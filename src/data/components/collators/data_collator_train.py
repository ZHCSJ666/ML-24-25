"""Shamelessly lifted from https://github.com/JetBrains-Research/commit_message_generation"""

from typing import List, Optional, Tuple

import torch

from .data_collator_base import DataCollatorBase
from ...types import Batch, SingleExample


class DataCollatorTrain(DataCollatorBase):
    """This class is used to construct batches out of lists of examples in training/validation
    setting.

    Format: `[BOS] message [EOS]`

    Attributes:
        shift_labels: True to mimic transformers' seq2seq models ids/labels construction logic, False otherwise
         (pass False for decoder class).
    """

    shift_labels: bool
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
        shift_labels: bool,
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
        self.shift_labels = shift_labels
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[SingleExample]) -> Batch:
        """Processes a list of examples into a BatchTrain object."""
        encoder_input_ids, encoder_attention_mask = self._process_encoder_input(examples=examples)

        decoder_input_ids, decoder_attention_mask, labels = self._process_decoder_input(
            examples=examples
        )

        return BatchTrain(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            msg_input_ids=decoder_input_ids,
            msg_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _shift_for_encoder_decoder(
        self, ids: List[List[int]], labels: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """This method mimics transformers logic of ids and labels for EncoderDecoderModel (or
        T5ForConditionalGeneration).

        Starting from transformers v4.12, loss is now calculated in EncoderDecoderModel, not in
        decoder class. Also, decoder input ids are created automatically based on labels: labels
        are shifted and -100 is replaced with pad token.
        """
        if self.decoder_start_token_id is None:
            ids = [[self.msg_bos_token_id]] + ids[:-1]
        else:
            ids = [[self.decoder_start_token_id]] + ids[:-1]
        return ids, labels

    def _process_decoder_input(
        self, examples: List[SingleExample]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares decoder input for train/validation:

          * constructs message input with special tokens
          * constructs labels
          * pads, converts to tensors

        Args:
            examples: A list of inputs for current batch.

        Returns:
            Tuple of three tensors: input ids, attention masks, labels.
        """
        message_inputs: List[List[int]] = [example.decoder_input_ids for example in examples]

        all_msg_ids: List[torch.Tensor] = []
        all_msg_masks: List[torch.Tensor] = []
        all_msg_labels: List[torch.Tensor] = []

        for message_ids in message_inputs:
            message_ids = message_ids[: self.decoder_context_max_len - 2]

            cur_ids = [[self.msg_bos_token_id]] + [message_ids] + [[self.msg_eos_token_id]]
            cur_labels = [[self.msg_bos_token_id]] + [message_ids] + [[self.msg_eos_token_id]]

            if self.shift_labels:
                cur_ids, cur_labels = self._shift_for_encoder_decoder(cur_ids, cur_labels)

            cur_ids_tensor = torch.tensor(
                [ex for sublist in cur_ids for ex in sublist], dtype=torch.int64
            )
            cur_labels_tensor = torch.tensor(
                [ex for sublist in cur_labels for ex in sublist], dtype=torch.int64
            )
            cur_mask_tensor = torch.ones_like(cur_ids_tensor)

            all_msg_ids.append(cur_ids_tensor)
            all_msg_masks.append(cur_mask_tensor)
            all_msg_labels.append(cur_labels_tensor)

        msg_max_len = max(len(tensor) for tensor in all_msg_ids)
        all_msg_ids = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=self.msg_pad_token_id,
                left=False,
            )
            for tensor in all_msg_ids
        ]
        all_msg_masks = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=0,
                left=False,
            )
            for tensor in all_msg_masks
        ]
        all_msg_labels = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=-100,
                left=False,
            )
            for tensor in all_msg_labels
        ]

        return (
            torch.stack(all_msg_ids),
            torch.stack(all_msg_masks),
            torch.stack(all_msg_labels),
        )
