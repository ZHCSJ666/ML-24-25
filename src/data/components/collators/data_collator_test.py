"""Shamelessly lifted from https://github.com/JetBrains-Research/commit_message_generation"""

from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerFast

from .data_collator_base import DataCollatorBase
from ...types import BatchTest, SingleExample


class DataCollatorTest(DataCollatorBase[BatchTest]):
    """This class is used to construct batches out of lists of examples in evaluation setting.

    We can emulate completion workflow by adding X% of characters of each message
    to decoder context.

    Format: `[BOS] X% characters of message`

    Attributes:
        context_ratio: (context_ratio * 100)% of characters of each message will
         be added to decoder context (should be a float between 0.0 and 1.0).
        max_new_tokens: A maximum number of generated tokens during generation.
    """

    diff_tokenizer: PreTrainedTokenizerFast
    msg_tokenizer: PreTrainedTokenizerFast
    context_ratio: float
    max_new_tokens: int = 15  # TODO: make configurable
    decoder_start_token_id: Optional[int] = None

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
        context_ratio: float,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        max_new_tokens: int = 15,
        decoder_start_token_id: Optional[int] = None,
    ):

        super().__init__(
            diff_bos_token_id,
            diff_eos_token_id,
            diff_pad_token_id,
            msg_bos_token_id,
            msg_eos_token_id,
            msg_pad_token_id,
            msg_sep_token_id,
            encoder_context_max_len,
            decoder_context_max_len,
            completion,
            split_ratio,
        )
        self.diff_tokenizer = diff_tokenizer
        self.msg_tokenizer = msg_tokenizer
        self.context_ratio = context_ratio
        self.max_new_tokens = max_new_tokens
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[SingleExample]) -> BatchTest:
        """Processes a list of examples into a BatchTest object."""
        encoder_input_ids, encoder_attention_mask = self._process_encoder_input(examples=examples)

        (
            decoder_input_ids,
            decoder_attention_mask,
            targets,
            prefixes,
        ) = self._process_decoder_input(examples=examples)

        return BatchTest(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=None,
            targets=targets,
            prefixes=prefixes,
        )

    def _process_msg_gen(
        self, message_ids: List[int], context_len: Optional[int] = None
    ) -> Tuple[List[int], str, str]:
        """Builds context and target for completion-style generation. The last word in context is
        treated as prefix for the first generated word.

        Args:
            message_ids: Input message, tokenized.

        Returns: A tuple of length three, where
          - first element is the model input,
          - second element is the target,
          - third element is the prefix.
        """
        # context_ratio = 0.0 => do not include anything in context
        if self.context_ratio == 0.0:
            return [], self.msg_tokenizer.decode(message_ids, skip_special_tokens=True), ""  # type: ignore[attr-defined]

        # context_ratio = 1.0 => include the whole message in context
        if self.context_ratio == 1.0:
            return message_ids, "", ""

        message = self.msg_tokenizer.decode(message_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        if not context_len:
            assert self.context_ratio
            context_len = int(len(message) * self.context_ratio)
        input, target = message[:context_len], message[context_len:]

        # if context is empty, use the whole message as target
        # (might happen with very short messages and small context_ratio)
        if not input:
            return [], target, ""

        # if the last word in context is full, do not use prefix
        if input[-1].isspace():
            context = input
            prefix = ""
        else:
            context, prefix = " ".join(input.split()[:-1]), input.split()[-1]

            if len(context) > 0:
                prefix = " " + prefix

        return self.msg_tokenizer(context, add_special_tokens=False).input_ids, target, prefix  # type: ignore[operator]

    def _process_decoder_input(
        self, examples: List[SingleExample]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """Process the input examples into decoder input on evaluation stage.

        The input examples are processed as follows:
            * The message input ids for each example are extracted.
            * Messages are processed for generation according to context ratio configuration.
            * Inputs are padded to the maximum length in the batch and converted to tensors.

        Args:
            examples: A list of input examples to process.

        Returns:
            A tuple containing:
                A tensor of shape (batch_size, seq_len) representing the input ids for the decoder.
                A tensor of shape (batch_size, seq_len) representing the attention masks for the decoder.
                A list of target strings for each example.
                A list of prefix strings for each example.
        """
        message_inputs: List[List[int]] = [example.msg_input_ids for example in examples]

        all_msg_ids: List[torch.Tensor] = []
        all_msg_masks: List[torch.Tensor] = []

        all_msg_targets: List[str] = []
        all_msg_prefixes: List[str] = []

        for message_ids in message_inputs:
            message_ids = message_ids[: self.decoder_context_max_len - 1]
            message_ids, target, prefix = self._process_msg_gen(message_ids)

            if self.decoder_start_token_id is None:
                start_token_id = self.msg_bos_token_id
            else:
                start_token_id = self.decoder_start_token_id
            cur_ids = [[start_token_id]] + [message_ids]
            cur_ids_tensor = torch.tensor(
                [ex for sublist in cur_ids for ex in sublist], dtype=torch.int64
            )
            cur_mask_tensor = torch.ones_like(cur_ids_tensor)

            all_msg_ids.append(cur_ids_tensor)
            all_msg_masks.append(cur_mask_tensor)

            all_msg_targets.append(target)
            all_msg_prefixes.append(prefix)

        msg_max_len = max(len(tensor) for tensor in all_msg_ids)

        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        all_msg_ids = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=self.msg_pad_token_id,
                left=True,
            )
            for tensor in all_msg_ids
        ]
        all_msg_masks = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=0,
                left=True,
            )
            for tensor in all_msg_masks
        ]

        return (
            torch.stack(all_msg_ids),
            torch.stack(all_msg_masks),
            all_msg_targets,
            all_msg_prefixes,
        )
