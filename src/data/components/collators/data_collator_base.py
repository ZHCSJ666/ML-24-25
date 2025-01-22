"""Shamelessly lifted from https://github.com/JetBrains-Research/commit_message_generation"""

from typing import List, Dict, Any

import torch

from src.data.types import SingleExample, Batch


class DataCollatorBase:
    """Base class for utilities both for training and evaluation collators (e.g. processing encoder
    input).

    Attributes:
        msg_*_token_id: Corresponding special token for message tokenizer.
        diff_*_token_id: Corresponding special token for diff tokenizer.
        encoder_context_max_len: Maximum allowed number of tokens in encoder context.
        decoder_context_max_len: Maximum allowed number of tokens in decoder context.
    """

    completion: bool
    # Ratio of message to be used as prefix for chat completion
    split_ratio: float
    msg_bos_token_id: int
    msg_eos_token_id: int
    msg_pad_token_id: int
    msg_sep_token_id: int
    diff_bos_token_id: int
    diff_eos_token_id: int
    diff_pad_token_id: int
    encoder_context_max_len: int
    decoder_context_max_len: int

    def __init__(
        self,
        diff_bos_token_id: int,
        diff_eos_token_id: int,
        diff_pad_token_id: int,
        msg_bos_token_id: int,
        msg_eos_token_id: int,
        msg_pad_token_id: int,
        msg_sep_token_id: int,
        encoder_context_max_len: int,
        decoder_context_max_len: int,
        completion: bool,
        split_ratio: float,
    ) -> None:
        self.diff_bos_token_id = diff_bos_token_id
        self.diff_eos_token_id = diff_eos_token_id
        self.diff_pad_token_id = diff_pad_token_id
        self.msg_bos_token_id = msg_bos_token_id
        self.msg_eos_token_id = msg_eos_token_id
        self.msg_pad_token_id = msg_pad_token_id
        self.msg_sep_token_id = msg_sep_token_id
        self.encoder_context_max_len = encoder_context_max_len
        self.decoder_context_max_len = decoder_context_max_len
        self.completion = completion
        self.split_ratio = split_ratio

    def __call__(self, examples: List[SingleExample]) -> Batch:
        raise NotImplementedError()

    def _pad_tensor(
        self, input_tensor: torch.Tensor, pad_len: int, value: int, left: bool
    ) -> torch.Tensor:
        return torch.nn.functional.pad(
            input_tensor,
            pad=[pad_len, 0] if left else [0, pad_len],
            mode="constant",
            value=value,
        )

    def _process_inputs(self, inputs: List[List[int]], are_messages: bool = False):
        """This helper method processes either diffs or messages as encoder input.

        It truncates the inputs to the maximum allowed length.

        It also adds all required special tokens: format is [BOS] input [EOS].

        Finally, it is responsible for padding to maximum length in batch and conversion to torch.Tensor.

        Args:
            inputs: A list of tokenized examples from the current batch.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        if are_messages:
            bos_token_id = self.msg_bos_token_id
            eos_token_id = self.msg_eos_token_id
            pad_token_id = self.msg_pad_token_id
        else:
            bos_token_id = self.diff_bos_token_id
            eos_token_id = self.diff_eos_token_id
            pad_token_id = self.diff_pad_token_id

        inputs = [
            [bos_token_id] + example[: self.encoder_context_max_len - 2] + [eos_token_id]
            for example in inputs
        ]
        inputs_tensors = [torch.tensor(ids, dtype=torch.int64) for ids in inputs]
        masks_tensors = [torch.ones_like(ids) for ids in inputs_tensors]

        # pad tensors to max length in batch
        inputs_max_len = max(len(tensor) for tensor in inputs)
        inputs_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=inputs_max_len - tensor.numel(),
                value=pad_token_id,
                left=False,
            )
            for tensor in inputs_tensors
        ]
        masks_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=inputs_max_len - tensor.numel(),
                value=0,
                left=False,
            )
            for tensor in masks_tensors
        ]
        return torch.stack(inputs_tensors), torch.stack(masks_tensors)

    def _process_encoder_input(self, examples: List[SingleExample]):
        """Process encoder input (diffs and optionally partial messages)."""

        input_ids = [example.diff_input_ids for example in examples]

        if self.completion:
            msg_ids = [example["msg_input_ids"] for example in examples]
            for i, msg in enumerate(msg_ids):
                split_idx = int(len(msg) * self.split_ratio)
                partial_msg = msg[:split_idx]
                examples[i].diff_input_ids = msg[split_idx:]
                input_ids[i] = input_ids[i] + [self.msg_sep_token_id] + partial_msg

        return self._process_inputs(input_ids, are_messages=False)

    def _process_decoder_input(self, examples: List[Dict[str, Any]]):
        """Process decoder input (messages).
        In completion mode, only the remaining part of message is used as target."""
        input_ids = [example["msg_input_ids"] for example in examples]
        return self._process_inputs(input_ids, are_messages=True)
