from typing import Optional

import torch


class Batch:
    """
    Represents a batch of input data for training a model that generates Git commit messages from Git diffs.

    Attributes:
        input_ids (torch.Tensor):
            A tensor containing tokenized input sequences. The content of this tensor varies depending on the task:
            - **Sequence-to-sequence training**: Contains only Git diffs.
            - **Language modeling training**: Contains Git diffs followed by a separator token and commit messages.
            - **Language modeling inference**: Contains Git diffs followed by a separator token.

        attention_mask (torch.Tensor):
            A tensor indicating which positions in `input_ids` should be attended to, where non-zero values
            represent valid tokens.

        decoder_input_ids (Optional[torch.Tensor]):
            A tensor containing shifted tokenized commit message inputs for training. This field is specific to
            encoder-decoder models, and contains the input IDs that will be fed to the decoder. These inputs are usually
            built in a way specific to each model.

            Most encoder-decoder models (BART, T5) create their decoder_input_ids on their own from the labels.
            In such models, passing the labels is the preferred way to handle training.

        decoder_attention_mask (Optional[torch.Tensor]):
            A tensor indicating valid positions in `msg_input_ids`, ensuring the model attends only to meaningful tokens.
            This field is specific to encoder-decoder models

        labels (Optional[torch.Tensor]):
            A tensor containing expected output tokens for training, used as target values during loss computation.
            - For sequence-to-sequence tasks, this represents the ground-truth git commit messages, and should be set
            during training, validation and testing.
            - For language-modeling tasks, this represents the next token to be predicted (causal language modeling) for
            the masked tokens to be predicted (masked language modeling), and should be set during training/validation.

        targets (Optional[list[str]]):
            A text sequence representing ground-truth output. For sequence-to-sequence tasks, this is not required.
            This field is only used at test time for language modeling tasks.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_ids: Optional[torch.Tensor] = None
    decoder_attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    targets: Optional[list[str]] = None

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        targets: Optional[list[str]] = None,
    ) -> None:
        assert torch.is_tensor(input_ids)
        assert torch.is_tensor(attention_mask)
        assert decoder_input_ids is None == decoder_attention_mask is None
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels
        self.targets = targets

    def pin_memory(self) -> "Batch":
        """Pins all tensors to memory, making them GPU-accessible for faster data transfer if using
        a CUDA-capable device. This operation converts all available attributes to pinned memory,
        provided they are not None.

        Returns:
            self (Batch): The batch instance with all tensors pinned to memory.
        """
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        if self.decoder_input_ids is not None:
            self.decoder_input_ids = self.decoder_input_ids.pin_memory()
        if self.decoder_attention_mask is not None:
            self.decoder_attention_mask = self.decoder_attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, device: str | torch.device) -> "Batch":
        """
        Moves all tensors in the batch to the specified device.

        Args:
            device (str | torch.device): The target device (e.g., `"cuda"`, `"cpu"`, or a specific device index).

        Returns:
            Batch: The current instance with all tensors moved to the specified device.
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.decoder_input_ids is not None:
            self.decoder_input_ids = self.decoder_input_ids.to(device)
        if self.decoder_attention_mask is not None:
            self.decoder_attention_mask = self.decoder_attention_mask.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self
