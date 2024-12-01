"""Shamelessly lifted from https://github.com/JetBrains-Research/commit_message_generation"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from src.data.types import SingleExample


@dataclass
class BaseCollatorUtils:
    """Base class for utilities both for training and evaluation collators (e.g. processing encoder
    input).

    Attributes:
        msg_*_token_id: Corresponding special token for message tokenizer.
        diff_*_token_id: Corresponding special token for diff tokenizer.
        encoder_context_max_len: Maximum allowed number of tokens in encoder context.
        decoder_context_max_len: Maximum allowed number of tokens in decoder context.
        encoder_input_type: Should be `diff`, corresponding data will be used
          to construct encoder input.
    """

    msg_bos_token_id: int
    msg_eos_token_id: int
    msg_pad_token_id: int
    msg_sep_token_id: int
    diff_bos_token_id: int
    diff_eos_token_id: int
    diff_pad_token_id: int
    encoder_context_max_len: int
    decoder_context_max_len: int
    encoder_input_type: str

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

    def _process_encoder_input(
        self, examples: List[SingleExample]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A helper method to process encoder input.

        Only diff can be passed to encoder.

        Args:
            examples: A batch of examples from dataset.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        if self.encoder_input_type == "diff":
            diff_inputs: List[List[int]] = [example.diff_input_ids for example in examples]
            results = self._process_inputs(diff_inputs)
        else:
            raise ValueError("Unknown encoder input type")
        return results
