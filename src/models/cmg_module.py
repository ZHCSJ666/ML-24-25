from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torchmetrics import MetricCollection
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.data.types import Batch, BatchTest, BatchTrain
from src.metrics import MRR, Accuracy


class CommitMessageGenerationModule(LightningModule):
    """Git commit message generation module.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: nn.Module | Callable[..., nn.Module],
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler],
        compile: bool,
        shift: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a `CommitMessageGenerationModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model.
        :param shift: Applicable if the `net` is a decoder-only style model like GPT.
            For encoder-decoder architecture this should be False
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["net"] if isinstance(net, nn.Module) else []
        )
        self.criterion = nn.NLLLoss(ignore_index=-100)  # Updated to ignore padding tokens

        self.net = net

        # We'd add string-based metrics like BLEU, ROUGE later.
        # For now, we are using tensor-based metrics
        metrics = MetricCollection(
            {
                "acc_top1": Accuracy(top_k=1, shift=shift),
                "acc_top5": Accuracy(top_k=5, shift=shift),
                "MRR_top5": MRR(top_k=5, shift=shift),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.msg_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.diff_tokenizer: Optional[PreTrainedTokenizerFast] = None

    def forward(self, batch: Batch) -> Any:
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics.reset()

    def model_step(self, batch: BatchTrain, split: str) -> Dict[str, Any]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data.
        :param split: One of "train" or "val".
        :return: A dictionary containing loss and, if applicable, BLEU scores.
        """
        result = {}

        outputs = self.forward(batch)  # Shape: (batch, seq_len, vocab_size)
        if isinstance(outputs, Seq2SeqLMOutput):
            loss = outputs.loss
            logits = outputs.logits
        else:
            logits = outputs  # (batch * seq_len)
            # Compute loss
            loss = self.criterion(outputs.permute(0, 2, 1), batch.labels)
        result["loss"] = loss
        batch_size = len(batch.encoder_input_ids)
        self.log(
            f"{split}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        # metrics
        metric = getattr(self, f"{split}_metrics")
        metric(logits, batch.labels)
        self.log_dict(
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if self.global_step > 0 and self.current_epoch % 5 == 0:
            self.net.eval()
            predictions = self.generate(batch)
            # decode & postprocess data
            string_results = self._postprocess_generated(batch, predictions)
            self.log_results(f"{split}/", string_results)
            self.net.train()

            # TODO: Compute and log string based metrics like BLEU and ROUGE later here

        return result

    def training_step(self, batch: BatchTrain, batch_idx: int) -> Dict[str, Any]:
        """Training step.

        :param batch: A training batch.
        :param batch_idx: Index of the batch.
        :return: Dictionary containing the loss.
        """
        result = self.model_step(batch, "train")
        return {"loss": result["loss"]}

    def test_step(self, batch: BatchTest, batch_idx: int) -> None:
        """Test step.

        :param batch: A test batch.
        :param batch_idx: Index of the batch.
        """
        predictions = self.generate(batch)

        # decode & postprocess data
        string_results = self._postprocess_generated(batch, predictions)
        self.log_results("test/", string_results)

    def generate(self, batch: Batch, **kwargs) -> Any:
        kwargs = kwargs or self.hparams.generation_kwargs or {}
        return self.net.generate(
            batch,
            **kwargs,
            prefix_allowed_tokens_fn=None,
            pad_token_id=self.msg_tokenizer.pad_token_id,
            bos_token_id=self.msg_tokenizer.bos_token_id,
            eos_token_id=self.msg_tokenizer.eos_token_id,
        )

    def _postprocess_generated(
        self, batch: Batch, predictions: torch.Tensor
    ) -> List[Dict[str, str]]:
        """Decodes predictions and context.

        Args:
            batch: Model inputs.
            predictions: Model predictions.

        Returns:
            A dict with decoded sources/predictions.
        """
        decoded_inputs = self.decode_src(batch.encoder_input_ids, skip_special_tokens=True)[0]
        decoded_preds = self.decode_tgt(predictions, skip_special_tokens=True)[0]

        return [
            {
                "input": input_,
                "prediction": pred,
            }
            for input_, pred, in zip(decoded_inputs, decoded_preds)
        ]

    def decode_src(self, *args, **kwargs):
        return tuple(self.diff_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def decode_tgt(self, *args, **kwargs):
        return tuple(self.msg_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: BatchTrain, batch_idx: int) -> None:
        """Validation step.

        :param batch: A validation batch.
        :param batch_idx: Index of the batch.
        """
        self.model_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

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

        if self.msg_tokenizer is None or self.diff_tokenizer is None:
            datamodule = self.trainer.datamodule
            self.msg_tokenizer = datamodule.msg_tokenizer
            self.diff_tokenizer = datamodule.diff_tokenizer

        if not isinstance(self.net, nn.Module):
            self.net = self.hparams.net(
                encoder_vocab_size=self.diff_tokenizer.vocab_size,
                decoder_vocab_size=self.msg_tokenizer.vocab_size,
            )

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

    def log_results(self, prefix, results: List[Dict[str, str]], num_results: int = 1) -> None:
        """Log generated git commit message results.

        This method only supports TensorBoard at the moment.
        """
        tb_logger: Optional[TensorBoardLogger] = None
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break
        if tb_logger is None:
            return

        writer = tb_logger.experiment
        for result in results[:num_results]:
            for key, value in result.items():
                writer.add_text(prefix + key, value, self.global_step)


if __name__ == "__main__":
    _ = CommitMessageGenerationModule(None, None, None, None)
