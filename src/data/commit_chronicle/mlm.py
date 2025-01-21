import multiprocessing
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from datasets import load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from data.types import Batch
from src.data.components.collators.mlm import (
    compute_input_and_target_lengths,
    DataCollatorForT5MLM,
)
from src.data.commit_chronicle.preprocessor import CommitChroniclePreprocessor
from src.utils.more_utils import hash_dict
from loguru import logger


class CommitChronicleMLMDataModule(LightningDataModule):
    """Commit Chronicle dataset `LightningDataModule` for Masked Language Modeling (MLM).

    The Commit Chronicle dataset was introduced in the paper "From Commit Message Generation to History-Aware Commit Message Completion", ASE 2023.

    Read the docs:
        https://huggingface.co/datasets/JetBrains-Research/commit-chronicle
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        data_dir: str = "data/commit-chronicle",
        huggingface_path: str = "JetBrains-Research/commit-chronicle",
        languages: Sequence[str] = ("Go",),
        change_types: Sequence[str] = ("ADD",),
        input_max_len: int = 1024,
        mlm_probability: float = 0.15,
        mean_noise_span_length: float = 3.0,
        shift_labels: bool = True,
        line_sep: str = "\n",
        use_cache: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize a `CommitChronicleMLMDataModule`.

        Args:
            tokenizer: Tokenizer used to tokenize both the diffs and msgs.
            data_dir: Directory to save preprocessed data.
            languages: The languages to be included in the dataset.
            change_types: The change types to be included in the dataset.
                Each element should be one of 'ADD', 'DELETE', 'RENAME', 'COPY' or 'MODIFY'
            input_max_len: Maximum length for input git commit diff.
            line_sep: Newline separator used in data (should be the same for diffs and messages).
            use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.

            batch_size: The batch size. Defaults to `16`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.tokenizer = tokenizer
        self.processor = CommitChroniclePreprocessor(
            diff_line_sep=line_sep,
            change_types=change_types,
            languages=languages,
            huggingface_path=huggingface_path,
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # We increase the input_length, because instead of masking tokens T5 replaces
        # masked spans with a single token, therefore to avoid padding we need to have
        # longer sequences at the start, before masking
        before_mask_input_length, target_length = compute_input_and_target_lengths(
            inputs_length=input_max_len,
            noise_density=mlm_probability,
            mean_noise_span_length=mean_noise_span_length,
        )
        self.before_mask_input_length = before_mask_input_length
        self.target_length = target_length

        # noinspection PyTypeChecker
        self.train_val_collator = DataCollatorWrapper(
            DataCollatorForT5MLM(
                tokenizer=tokenizer,
                noise_density=mlm_probability,
                mean_noise_span_length=mean_noise_span_length,
                input_length=input_max_len,
                target_length=target_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
        self.test_collator = None

        self.batch_size_per_device = batch_size

    @property
    def processed_data_dir(self):
        config = {
            "path": self.hparams.huggingface_path,
            "change_types": self.hparams.change_types,
            "languages": self.hparams.languages,
            "line_sep": self.hparams.line_sep,
        }
        return Path(self.hparams.data_dir) / f"processed-{hash_dict(config)}"

    @property
    def tokenized_data_dir(self):
        config = {
            "path": self.hparams.huggingface_path,
            "change_types": self.hparams.change_types,
            "languages": self.hparams.languages,
            "line_sep": self.hparams.line_sep,
            "tokenizer": self.tokenizer.name_or_path,
            "input_max_len": self.hparams.input_max_len,
            "mlm_probability": self.hparams.mlm_probability,
            "mean_noise_span_length": self.hparams.mean_noise_span_length,
        }
        return Path(self.hparams.data_dir) / f"mlm-{hash_dict(config)}"

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        for split in ["train", "validation", "test"]:
            processed_path = self.processed_data_dir / split
            tokenized_path = self.tokenized_data_dir / split
            if not tokenized_path.exists():
                self.processor(
                    output_dir=processed_path,
                    split=split,
                    use_cache=self.hparams.use_cache,
                )
                dataset = load_from_disk(processed_path)
                dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    fn_kwargs={
                        "tokenizer": self.tokenizer,
                        "in_length": self.before_mask_input_length,
                    },
                    remove_columns=["diff", "msg", "repo"],
                    batch_size=2000,
                    num_proc=max(1, multiprocessing.cpu_count() - 1),
                )
                size = len(dataset)
                # remove examples that don't meet before_mask_input_length
                # TODO: find a better fix instead of just discarding data
                dataset = dataset.filter(
                    lambda x: len(x["input_ids"]) == self.before_mask_input_length,
                    num_proc=max(1, multiprocessing.cpu_count() - 1),
                )
                new_size = len(dataset)
                logger.info(f"Dataset size truncated from {size} to {new_size}")
                dataset.save_to_disk(tokenized_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        def create_dataset(split):
            return load_from_disk(self.tokenized_data_dir / split)

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = create_dataset("train")
            self.data_val = create_dataset("validation")
            self.data_test = create_dataset("test")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.train_val_collator,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.train_val_collator,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.train_val_collator,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )


class DataCollatorWrapper:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, batch):
        targets = (
            [example.pop("target", None) for example in batch]
            if batch[0].get("target") is not None
            else None
        )
        outputs = self.collator(batch)
        outputs["target"] = targets
        return Batch(
            input_ids=outputs["input_ids"],
            attention_mask=outputs["attention_mask"],
            labels=outputs["labels"].long(),
            targets=targets,
        )


def tokenize_function(examples, tokenizer: PreTrainedTokenizerFast, in_length):
    tokenizer_out = tokenizer(text=examples["diff"], return_attention_mask=False)

    input_ids = tokenizer_out["input_ids"]

    concatenated_ids = np.concatenate(input_ids)

    # batch_size = len(input_ids)
    # total_length = min(batch_size * in_length, concatenated_ids.shape[0])
    # total_length = (total_length // batch_size) * batch_size
    #
    # concatenated_ids = concatenated_ids[:total_length]
    # concatenated_ids = concatenated_ids.reshape(batch_size, -1)
    # result = {"input_ids": concatenated_ids}

    batch_size = len(input_ids)
    total_length = concatenated_ids.shape[
        0
    ]  # min(batch_size * in_length, concatenated_ids.shape[0])
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length]
    concatenated_ids = concatenated_ids.reshape(-1, in_length)
    result = {"input_ids": concatenated_ids}

    return result
