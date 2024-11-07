from dataclasses import dataclass
from typing import List, Optional

import torch


@torch.jit.script
class SingleExample:
    """
    A class to represent a single example for a Git commit message generation project.

    This class stores information related to a Git diff input, commit message, and commit history
    which are used as input features for a machine learning model in a Git commit message generation
    task.

    Attributes:
    ----------
    diff_input_ids : List[int]
        List of integer tokens representing the current Git diff input.
    msg_input_ids : List[int]
        List of integer tokens representing the Git commit message.
    history_input_ids : List[List[int]]
        A list of lists, where each inner list represents the tokenized history of previous
        Git diffs or commits.
    """

    diff_input_ids: List[int]
    msg_input_ids: List[int]
    history_input_ids: List[List[int]]
    pos_in_file: int

    def __init__(
        self,
        diff_input_ids: List[int],
        msg_input_ids: List[int],
        history_input_ids: List[List[int]],
        pos_in_file: int = -1,
    ) -> None:
        self.diff_input_ids = diff_input_ids
        self.msg_input_ids = msg_input_ids
        self.history_input_ids = history_input_ids
        self.pos_in_file = pos_in_file


@dataclass
class Batch:
    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    retrieved_diff_input_ids: Optional[torch.Tensor]
    retrieved_diff_attention_mask: Optional[torch.Tensor]
    retrieved_msg_input_ids: Optional[torch.Tensor]
    retrieved_msg_attention_mask: Optional[torch.Tensor]

    def pin_memory(self):
        self.encoder_input_ids = self.encoder_input_ids.pin_memory()
        self.encoder_attention_mask = self.encoder_attention_mask.pin_memory()
        self.decoder_input_ids = self.decoder_input_ids.pin_memory()
        self.decoder_attention_mask = self.decoder_attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        if self.retrieved_diff_input_ids is not None:
            self.retrieved_diff_input_ids = self.retrieved_diff_input_ids.pin_memory()
        if self.retrieved_diff_attention_mask is not None:
            self.retrieved_diff_attention_mask = (
                self.retrieved_diff_attention_mask.pin_memory()
            )
        if self.retrieved_msg_input_ids is not None:
            self.retrieved_msg_input_ids = self.retrieved_msg_input_ids.pin_memory()
        if self.retrieved_msg_attention_mask is not None:
            self.retrieved_msg_attention_mask = (
                self.retrieved_msg_attention_mask.pin_memory()
            )
        return self


@dataclass
class BatchTrain(Batch):
    labels: torch.Tensor


@dataclass
class BatchTest(Batch):
    targets: List[str]
    prefixes: List[str]
