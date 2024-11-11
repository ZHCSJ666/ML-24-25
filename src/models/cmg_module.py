from typing import Any, Dict

import torch
from lightning import LightningModule
import torch.nn as nn
from data.components.tokenization import load_tokenizers
from data.types import Batch, BatchTest, BatchTrain
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from torchmetrics import BLEUScore

class CommitMessageGenerationModule(LightningModule):
    """Git commit message generation module.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        
    ) -> None:
        """Initialize a `EncoderDecoderLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(logger=False)

        # Define the loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Updated to ignore padding tokens

        # Assign the model
        self.net = net

        # Assign the tokenizer for decoding
        self.tokenizer = tokenizer

        # Initialize BLEU score metrics for validation and testing
        self.val_bleu = BLEUScore()
        self.test_bleu = BLEUScore()

    def forward(self, batch: Batch) -> Any:
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch: Batch, split: str) -> Dict[str, Any]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data.
        :param split: One of "train", "val", or "test".
        :return: A dictionary containing loss and, if applicable, BLEU scores.
        """
        # Forward pass
        outputs = self.forward(batch)  # Shape: (batch, seq_len, vocab_size)

        result = {}

        if isinstance(batch, BatchTrain):
            # Reshape outputs and labels for loss computation
            logits = outputs.view(-1, outputs.size(-1))  # (batch * seq_len, vocab_size)
            labels = batch.labels.view(-1)  # (batch * seq_len)

            # Compute loss
            loss = self.criterion(logits, labels)
            result["loss"] = loss

            if split == "val":
                # Compute BLEU score
                preds = torch.argmax(outputs, dim=-1)  # (batch, seq_len)
                preds = preds.cpu().numpy().tolist()
                targets = batch.labels.cpu().numpy().tolist()

                # Decode predictions and targets
                decoded_preds = [self.tokenizer_decode(p) for p in preds]
                decoded_targets = [self.tokenizer_decode(t) for t in targets]

                # Update BLEU score
                bleu = self.val_bleu(decoded_preds, decoded_targets)
                result["val_bleu"] = bleu

        elif isinstance(batch, BatchTest):
            # During testing, generate commit messages and compute BLEU scores
            generated_ids = self.net.generate(
                src=batch,
                max_length=self.hparams.max_length if hasattr(self.hparams, 'max_length') else 512,
                num_beams=self.hparams.num_beams if hasattr(self.hparams, 'num_beams') else 5,
                early_stopping=True,
            )

            # Decode generated sequences
            decoded_preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids.cpu().numpy().tolist()]
            decoded_targets = batch.targets  # List[str]

            # Compute BLEU score
            bleu = self.test_bleu(decoded_preds, decoded_targets)
            result["test_bleu"] = bleu

        return result

    def training_step(self, batch: BatchTrain, batch_idx: int) -> Dict[str, Any]:
        """Training step.

        :param batch: A training batch.
        :param batch_idx: Index of the batch.
        :return: Dictionary containing the loss.
        """
        result = self.model_step(batch, "train")
        loss = result["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def test_step(self, batch: BatchTest, batch_idx: int) -> None:
        """Test step.

        :param batch: A test batch.
        :param batch_idx: Index of the batch.
        """
        result = self.model_step(batch, "test")
        bleu = result["test_bleu"]
        self.log("test_bleu", bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: BatchTrain, batch_idx: int) -> None:
        """Validation step.

        :param batch: A validation batch.
        :param batch_idx: Index of the batch.
        """
        result = self.model_step(batch, "val")
        loss = result["loss"]
        bleu = result["val_bleu"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bleu", bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: BatchTest, batch_idx: int) -> None:
        self.model_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CommitMessageGenerationModule(None, None, None, None)
