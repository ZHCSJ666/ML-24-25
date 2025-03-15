import logging

from lightning import LightningModule
from torchmetrics import MetricCollection, SumMetric
import os
from loguru import logger
import json

from src.metrics.bleu import SacreBLEUScore
from src.metrics.rouge import ROUGEScore
from src.utils.chat_completion import LLMChatCompleter, LLMChatCompleterResponse
from src.utils.more_utils import TextLoggingMixin

# https://stackoverflow.com/a/53014308/7121776
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


logging.getLogger("openai").setLevel(logging.WARNING)
# Suppress urllib3 logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress httpx logs (if using httpx)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMApiCommitMessageGenerationModule(LightningModule, TextLoggingMixin):
    """Git commit message generation module.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, completer: LLMChatCompleter) -> None:
        """Initialize a `CommitMessageGenerationModule`.

        :param completer: LLMChatCompleter
        """
        super().__init__()

        self.completer = completer
        self.test_text_metrics = MetricCollection(
            {
                "sacre_bleu1": SacreBLEUScore(n_gram=1),
                "sacre_bleu4": SacreBLEUScore(n_gram=4),
                "rouge1": ROUGEScore(rouge_keys="rouge1")["rouge1_fmeasure"],
                "rouge2": ROUGEScore(rouge_keys="rouge2")["rouge2_fmeasure"],
                "rougeL": ROUGEScore(rouge_keys="rougeL")["rougeL_fmeasure"],
                "rougeLsum": ROUGEScore(rouge_keys="rougeLsum")["rougeLsum_fmeasure"],
            },
            prefix="test/",
        )
        self.test_actual_prompt_token_count_metric = MetricCollection(
            {
                "actual_prompt_token_count": SumMetric(),
            },
            prefix="test/",
        )
        self.test_estimated_prompt_token_count_metric = MetricCollection(
            {
                "estimated_prompt_token_count": SumMetric(),
            },
            prefix="test/",
        )
        self.string_results = []


    def forward(self, batch: list[dict[str, str]]) -> list[LLMChatCompleterResponse]:
        """Forward pass of the model."""
        results = []
        for item in batch:
            results.append(
                self.completer.complete_chat(
                    [
                        {
                            "role": "system",
                            "content": item["system_content"],
                        },
                        {
                            "role": "user",
                            "content": item["user_content"],
                        },
                    ]
                )
            )
        return results

    def test_step(self, batch: list[dict[str, str]], batch_idx: int) -> None:
        """Test step.

        :param batch: A test batch.
        :param batch_idx: Index of the batch.
        """
        if batch_idx < self.start_batch_idx:
            return
        elif batch_idx == self.start_batch_idx:
            logger.info(f"Resuming from {batch_idx}")
        batch_size = len(batch)
        messages = [item["msg"] for item in batch]
        outputs = self.forward(batch)
        predictions = [output.content for output in outputs]

        # text metrics
        self.test_text_metrics(predictions, messages)
        self.log_dict(
            self.test_text_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # text output
        for idx in range(batch_size):
            self.string_results.append(
                {
                    "input": batch[idx]["diff"],
                    "prediction": outputs[idx].content,
                    "target": batch[idx]["msg"],
                }
            )

        # token count metrics
        actual_prompt_token_count = []
        estimated_prompt_token_count = []

        for item, output in zip(batch, outputs):
            actual_prompt_token_count.append(output.prompt_token_count)
            estimated_prompt_token_count.append(
                item["token_stats"]["truncated"]["num_total_tokens"]
            )
        self.test_actual_prompt_token_count_metric(actual_prompt_token_count)
        self.log_dict(
            self.test_actual_prompt_token_count_metric,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.test_estimated_prompt_token_count_metric(estimated_prompt_token_count)
        self.log_dict(
            self.test_estimated_prompt_token_count_metric,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )

    def on_test_epoch_end(self) -> None:
        self.log_text("test/", self.string_results)

    def configure_optimizers(self) -> None:
        """Return optimizers"""
        return None
