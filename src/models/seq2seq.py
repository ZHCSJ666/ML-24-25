import random
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MetricCollection, SumMetric
from transformers import PreTrainedTokenizerFast
from loguru import logger

from src.data.types import Batch
from src.metrics import MRR, Accuracy
from src.metrics.bleu import SacreBLEUScore
from src.metrics.rouge import ROUGEScore
from src.models.components.encoder_decoder import EncoderDecoder
from src.utils.more_utils import TextLoggingMixin


class Seq2SeqCommitMessageGenerationModule(LightningModule, TextLoggingMixin):
    """Git commit message generation module.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: nn.Module | Callable[..., nn.Module],
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler]] = None,
        compile: bool = False,
        shift: bool = False,
        pretrained: Optional[str] = None,
        val_text_metrics_every_step: bool = False,
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
        if pretrained is not None:
            checkpoint = torch.load(pretrained, map_location="cpu")
            state_dict = {k.replace("net.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.net.load_state_dict(state_dict)

        text_metrics = MetricCollection(
            {
                "sacre_bleu1": SacreBLEUScore(n_gram=1),
                "sacre_bleu4": SacreBLEUScore(n_gram=4),
                "rouge1": ROUGEScore(rouge_keys="rouge1")["rouge1_fmeasure"],
                "rouge2": ROUGEScore(rouge_keys="rouge2")["rouge2_fmeasure"],
                "rougeL": ROUGEScore(rouge_keys="rougeL")["rougeL_fmeasure"],
                "rougeLsum": ROUGEScore(rouge_keys="rougeLsum")["rougeLsum_fmeasure"],
            }
        )
        self.train_text_metrics = text_metrics.clone(prefix="train/")
        self.val_text_metrics = text_metrics.clone(prefix="val/")
        self.test_text_metrics = text_metrics.clone(prefix="test/")

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

        token_metrics = MetricCollection(
            {
                "num_processed_tokens": SumMetric(),
            }
        )
        self.train_token_metrics = token_metrics.clone(prefix="train/")
        self.val_token_metrics = token_metrics.clone(prefix="val/")
        self.test_token_metrics = token_metrics.clone(prefix="test/")

        self.msg_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.diff_tokenizer: Optional[PreTrainedTokenizerFast] = None

        # Used to store temp batch data to be used for logging at the end of epochs
        self.train_batch: Optional[Batch] = None
        self.val_batch: Optional[Batch] = None

    def forward(self, batch: Batch) -> Any:
        """Forward pass of the model.

        Args:
            batch: Input batch containing source and target sequences.

        Returns:
            dict: Dictionary containing model outputs including logits and predictions.
        """
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics.reset()
        self.val_batch = None
        self.train_token_metrics.reset()
        self.val_token_metrics.reset()
        self.test_token_metrics.reset()

    def common_step(self, batch: Batch, split: str) -> Optional[torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data.
        :param split: One of "train" or "val".
        :return: A dictionary containing loss and, if applicable, BLEU scores.
        """

        outputs = self.forward(batch)
        logits, loss = outputs["logits"], outputs.get("loss", None)
        if loss is None:
            loss = self.criterion(logits.permute(0, 2, 1), batch.labels)

        if not is_valid_tensor(logits) or not is_valid_tensor(loss):
            logger.warning(
                f"Any NaNs in: inputs={torch.isnan(batch.input_ids).any()}, logits={torch.isnan(logits).any()}, loss={torch.isnan(loss).any()}"
            )
            logger.warning(
                f"Any Infs in: inputs={torch.isinf(batch.input_ids).any()}, logits={torch.isinf(logits).any()}, loss={torch.isinf(loss).any()}"
            )
            return None

        batch_size = len(batch.input_ids)
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

        # token metrics
        num_tokens_processed = batch.attention_mask.sum().item()
        metric = getattr(self, f"{split}_token_metrics")
        metric(num_tokens_processed)
        self.log_dict(
            metric.compute(),  # don't pass the metric object to prevent lightning from resetting it here
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=1,
        )

        if self.hparams.val_text_metrics_every_step and split in ["val"]:
            self.evaluate_text(batch, split, log_text=False)

        # Here, we basically randomly save a batch
        # When epoch ends, we'll run this batch data through our model to generate text (in inference mode).
        # This is solely used for logging/visualization, so that we know how the model is currently generating text.
        saved_batch = getattr(self, f"{split}_batch")
        if saved_batch is None or random.random() > 0.5:  # nosec B311
            setattr(self, f"{split}_batch", batch)

        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Training step.

        :param batch: A training batch.
        :param batch_idx: Index of the batch.
        :return: Dictionary containing the loss.
        """
        return self.common_step(batch, "train")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Test step.

        :param batch: A test batch.
        :param batch_idx: Index of the batch.
        """
        self.evaluate_text(batch, "test")
        num_tokens_processed = batch.attention_mask.sum().item()
        self.test_token_metrics(num_tokens_processed)
        self.log_dict(
            self.test_token_metrics.compute(),  # don't pass the metric object to prevent lightning from resetting it here
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=1,
        )

    def on_test_epoch_end(self) -> None:
        self.test_token_metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.evaluate_text(self.train_batch, "train")
        self.train_batch = None
        self.train_token_metrics.reset()

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Validation step.

        :param batch: A validation batch.
        :param batch_idx: Index of the batch.
        """
        self.common_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # if we are not using text metrics on every step, then we can do it here
        if not self.hparams.val_text_metrics_every_step:
            self.evaluate_text(self.val_batch, "val")
        self.val_batch = None
        self.val_token_metrics.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.msg_tokenizer is None or self.diff_tokenizer is None:
            # noinspection PyUnresolvedReferences
            datamodule = self.trainer.datamodule
            self.msg_tokenizer = datamodule.msg_tokenizer
            self.diff_tokenizer = datamodule.diff_tokenizer
            datamodule.net = self.net.model

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
        try:
            optimizer = self.hparams.optimizer(self.trainer.model.parameters())
        except AttributeError:
            # to make optimizer creation with src.optimizers.create_optimizer work
            optimizer = self.hparams.optimizer(self.trainer.model)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                max_decay_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def evaluate_text(self, batch: Batch, split, log_text: bool = True) -> None:
        """Common operations to perform at the end of each epoch.

        Args:
            batch:
            split: String indicating the current stage ('train', 'val', or 'test').
            log_text:

        Returns:
            dict: Dictionary containing aggregated metrics for the epoch.
        """

        if batch is None:
            return
        is_train = self.net.training
        if is_train:
            self.net.eval()

        # obtain metric object
        text_metric: MetricCollection = getattr(self, f"{split}_text_metrics")

        # generate predictions
        predictions = self.generate(batch)

        # decode & postprocess data & log results
        string_results = self._postprocess_generated(batch, predictions)
        if log_text:
            self.log_text(text_metric.prefix, string_results)

        # compute metrics
        predictions, targets = zip(
            *[(result["prediction"], [result["target"]]) for result in string_results]
        )
        text_metric(predictions, targets)
        self.log_dict(
            text_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.input_ids),
        )

        if is_train:
            self.net.train()

    def generate(self, batch: Batch, **kwargs) -> Any:
        """Generate commit messages for given source sequences.

        Args:
            batch: Input batch containing source and target sequences.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            torch.Tensor: Generated sequence token IDs.
        """
        kwargs = kwargs or self.hparams.generation_kwargs or {}

        if isinstance(self.net, EncoderDecoder):
            return self.net.generate(batch)
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
        decoded_inputs = self.decode_src(batch.input_ids, skip_special_tokens=True)[0]
        decoded_preds = self.decode_tgt(predictions, skip_special_tokens=True)[0]

        if batch.labels is not None:
            targets = batch.labels.clone()
            targets = torch.where(targets == -100, self.msg_tokenizer.pad_token_id, targets)
            decoded_targets = self.decode_tgt(targets, skip_special_tokens=True)[0]
        else:
            decoded_targets = [None for _ in decoded_preds]

        results = []

        for (
            input_,
            pred,
            target,
        ) in zip(decoded_inputs, decoded_preds, decoded_targets):
            item = {
                "input": input_,
                "prediction": pred

            }
            if target is not None:
                item["target"] = target
            results.append(item)
        return results

    def decode_src(self, *args, **kwargs):
        """Decode source sequence IDs back to text.

        Args:
            ids: Tensor of token IDs representing the source sequence.

        Returns:
            str: Decoded source text.
        """
        return tuple(self.diff_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def decode_tgt(self, *args, **kwargs):
        """Decode target sequence IDs back to text.

        Args:
            ids: Tensor of token IDs representing the target sequence.

        Returns:
            str: Decoded target text.
        """
        return tuple(self.msg_tokenizer.batch_decode(arg, **kwargs) for arg in args)


def is_valid_tensor(tensor):
    return not torch.any(torch.isnan(tensor) | torch.isinf(tensor))
