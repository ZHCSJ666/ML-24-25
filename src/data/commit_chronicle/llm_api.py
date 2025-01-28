import multiprocessing
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets import load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.commit_chronicle.preprocessor import CommitChroniclePreprocessor
from src.utils.chat_completion import LLMChatCompleter
from src.utils.more_utils import hash_dict

from loguru import logger


class CommitChronicleLLMApiDataModule(LightningDataModule):
    """Commit Chronicle dataset `LightningDataModule` strictly for calls LLM-based APIs like OpenAI's, Ollama.

    The Commit Chronicle dataset was introduced in the paper "From Commit Message Generation to History-Aware Commit Message Completion", ASE 2023.

    Read the docs:
        https://huggingface.co/datasets/JetBrains-Research/commit-chronicle
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        system_content: str,
        user_content_template: str,
        max_prompt_token_count: int,
        completer: LLMChatCompleter,
        data_dir: str = "data/commit-chronicle",
        huggingface_path: str = "JetBrains-Research/commit-chronicle",
        languages: Sequence[str] = ("Go",),
        change_types: Sequence[str] = ("ADD",),
        line_sep: str = "\n",
        use_cache: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        print_token_stats: bool = True,
    ) -> None:
        """Initialize a `CommitChronicleCLMDataModule`.

        Args:
            data_dir: Directory to save preprocessed data.
            languages: The languages to be included in the dataset.
            change_types: The change types to be included in the dataset.
                Each element should be one of 'ADD', 'DELETE', 'RENAME', 'COPY' or 'MODIFY'
            line_sep: Newline separator used in data (should be the same for diffs and messages).
            use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.

            batch_size: The batch size. Defaults to `16`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
            huggingface_path:
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.completer = completer
        self.processor = CommitChroniclePreprocessor(
            diff_line_sep=line_sep,
            change_types=change_types,
            languages=languages,
            huggingface_path=huggingface_path,
        )
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @cached_property
    def processed_data_dir(self):
        config = {
            "path": self.hparams.huggingface_path,
            "change_types": self.hparams.change_types,
            "languages": self.hparams.languages,
            "line_sep": self.hparams.line_sep,
        }
        return Path(self.hparams.data_dir) / f"processed-{hash_dict(config)}"

    @cached_property
    def post_processed_data_dir(self):
        config = {
            "path": self.hparams.huggingface_path,
            "change_types": self.hparams.change_types,
            "languages": self.hparams.languages,
            "line_sep": self.hparams.line_sep,
            "system_content": self.hparams.system_content,
            "user_content_template": self.hparams.user_content_template,
            "max_prompt_token_count": self.hparams.max_prompt_token_count,
        }
        return Path(self.hparams.data_dir) / f"llm-api-{hash_dict(config)}"

    def prepare_data(self) -> None:
        """Prepares the dataset by processing and tokenizing data for training, validation, and testing splits.
        Note: Do not use it to assign state (self.x = y).
        """

        for split in ["test"]:
            processed_path = self.processed_data_dir / split
            post_processed_path = self.post_processed_data_dir / split
            if not post_processed_path.exists():
                self.processor(
                    output_dir=processed_path,
                    split=split,
                    use_cache=self.hparams.use_cache,
                )
                dataset = load_from_disk(processed_path)
                dataset = dataset.map(
                    map_example_to_request,
                    fn_kwargs={
                        "system_content": self.hparams.system_content,
                        "user_content_template": self.hparams.user_content_template,
                        "max_prompt_token_count": self.hparams.max_prompt_token_count,
                        "completer": self.completer,
                        "return_token_stats": True,
                    },
                    batched=False,
                    num_proc=max(1, multiprocessing.cpu_count() - 1),
                ).select_columns(["diff", "msg", "system_content", "user_content", "token_stats"])
                dataset.save_to_disk(post_processed_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to set up. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        def _load_dataset(split):
            return load_from_disk(self.post_processed_data_dir / split)

        # load and split datasets only if not loaded already
        if not self.data_test:
            self.data_test = _load_dataset("test")
            if self.hparams.print_token_stats:
                total_tokens = 0
                for item in self.data_test:
                    total_tokens += item["token_stats"]["truncated"]["num_total_tokens"]
                logger.info(
                    f"Total tokens in dataset: {total_tokens} Tokens per sample: {total_tokens / len(self.data_test)}"
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        raise NotImplementedError()

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=NoOpDataCollator(),
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )


class NoOpDataCollator:
    def __call__(self, examples):
        return examples


def map_example_to_request(
    example,
    completer: LLMChatCompleter,
    system_content: str,
    user_content_template: str,
    max_prompt_token_count: int,
    return_token_stats: bool = False,
) -> dict[str, Any]:
    validate_user_content_template(user_content_template)

    diff = example["diff"]

    non_diff_token_count = completer.count_tokens(
        [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content_template.format(diff=""),
            },
        ],
    )

    max_diff_token_count = max_prompt_token_count - non_diff_token_count
    assert (
        max_diff_token_count > 0
    ), f"Max diff token count must be more than zero, num_other_tokens={non_diff_token_count}"

    # truncate diff to `max_diff_token_count`
    diff_tokens = completer.encode(diff)
    orig_diff_token_count = len(diff_tokens)
    diff_tokens = diff_tokens[:max_diff_token_count]
    diff = completer.decode(diff_tokens)

    output = {
        "system_content": system_content,
        "user_content": user_content_template.format(diff=diff),
        "diff": diff,
    }
    if return_token_stats:
        output["token_stats"] = {
            "original": {
                "num_diff_tokens": orig_diff_token_count,
                "num_non_diff_tokens": non_diff_token_count,
                "num_total_tokens": orig_diff_token_count + non_diff_token_count,
            },
            "truncated": {
                "num_diff_tokens": len(diff_tokens),
                "num_non_diff_msg_tokens": non_diff_token_count,
                "num_total_tokens": len(diff_tokens) + non_diff_token_count,
            },
        }
    return output


def validate_user_content_template(message_template: str):
    assert "{diff}" in message_template
