from dataclasses import dataclass
from typing import List, Optional

import torch


@torch.jit.script
class SingleExample:
    """A class to represent a single example for a Git commit message generation project.

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
    """Represents a batch of data inputs for training a model designed to generate Git commit
    messages from Git diffs.

    Attributes:
        encoder_input_ids (torch.Tensor): Tensor containing tokenized input sequences for the encoder.
        encoder_attention_mask (torch.Tensor): Tensor representing the attention mask for the encoder input,
                                               where non-zero values indicate positions with valid tokens.
        decoder_input_ids (torch.Tensor): Tensor containing tokenized input sequences for the decoder, typically
                                          containing the start token or previous token when generating output.
        decoder_attention_mask (torch.Tensor): Tensor representing the attention mask for the decoder input,
                                               used to focus on valid tokens in the decoder input sequence.
        labels (Optional[torch.Tensor]): Tensor containing the target output tokens (labels) for training;
                                         these are compared with the model’s predicted output during loss calculation.
    """

    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]

    def pin_memory(self):
        """Pins all tensors to memory, making them GPU-accessible for faster data transfer if using
        a CUDA-capable device. This operation converts all available attributes to pinned memory,
        provided they are not None.

        Returns:
            self (Batch): The batch instance with all tensors pinned to memory.
        """
        self.encoder_input_ids = self.encoder_input_ids.pin_memory()
        self.encoder_attention_mask = self.encoder_attention_mask.pin_memory()
        self.decoder_input_ids = self.decoder_input_ids.pin_memory()
        self.decoder_attention_mask = self.decoder_attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self


@dataclass
class BatchTrain(Batch):
    """Represents a batch of training data .

    Attributes:
        labels (torch.Tensor): Tensor containing the target output tokens (labels) for training.
                               These labels are compared with the model’s predicted output to compute loss
                               during training. Unlike the optional `labels` attribute in `Batch`, `labels`
                               in `BatchTrain` are required.
    """

    labels: torch.Tensor


@dataclass
class BatchTest(Batch):
    targets: List[str]
    prefixes: List[str]
