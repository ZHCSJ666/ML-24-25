import multiprocessing
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets import load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForSeq2Seq

from src.data.commit_chronicle.preprocessor import CommitChroniclePreprocessor
from src.data.types import Batch
from src.utils.more_utils import hash_dict


class CommitChronicleSeq2SeqDataModule(LightningDataModule):
    """Commit Chronicle dataset `LightningDataModule` for commit message generation posed as a sequence-to-sequence task.

    The Commit Chronicle dataset was introduced in the paper "From Commit Message Generation to History-Aware Commit Message Completion", ASE 2023.

    References:
        https://huggingface.co/datasets/JetBrains-Research/commit-chronicle
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
        https://huggingface.co/learn/nlp-course/en/chapter7/5?fw=pt
    """

    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        data_dir: str = "data/commit-chronicle",
        huggingface_path: str = "JetBrains-Research/commit-chronicle",
        languages: Sequence[str] = ("Go",),
        change_types: Sequence[str] = ("ADD",),
        diff_max_len: int = 512,
        msg_max_len: int = 512,
        line_sep: str = "\n",
        use_cache: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize a `CommitChronicleCMGDataModule`.

        Args:
            diff_tokenizer: Tokenizer used to tokenize the diffs.
            msg_tokenizer: Tokenizer used to tokenize the commit messages.
            huggingface_path: Huggingface dataset path. Defaults to `JetBrains-Research/commit-chronicle`.
            data_dir: Directory to save preprocessed data.
            languages: The languages to be included in the dataset.
            change_types: The change types to be included in the dataset.
                Each element should be one of 'ADD', 'DELETE', 'RENAME', 'COPY' or 'MODIFY'
            diff_max_len: Maximum length for input git commit diff.
            msg_max_len: Maximum length for target git commit message.
            line_sep: Newline separator used in data (should be the same for diffs and messages).
            use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.
            batch_size: The batch size. Defaults to `16`. Needs to be divisible by the number of devices (e.g., if in a distributed setup).
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.diff_tokenizer = diff_tokenizer
        self.msg_tokenizer = msg_tokenizer
        self.processor = CommitChroniclePreprocessor(
            diff_line_sep=line_sep,
            change_types=change_types,
            languages=languages,
            huggingface_path=huggingface_path,
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.net = None
        self._train_val_collator = None
        self._test_collator = None

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
    def tokenized_data_dir(self):
        config = {
            "path": self.hparams.huggingface_path,
            "change_types": self.hparams.change_types,
            "languages": self.hparams.languages,
            "line_sep": self.hparams.line_sep,
            "diff_tokenizer": self.diff_tokenizer.name_or_path,
            "msg_tokenizer": self.msg_tokenizer.name_or_path,
            "diff_max_len": self.hparams.diff_max_len,
            "msg_max_len": self.hparams.msg_max_len,
        }
        return Path(self.hparams.data_dir) / f"seq2seq-{hash_dict(config)}"

    @property
    def train_val_collator(self):
        if self._train_val_collator is None:
            self._train_val_collator = DataCollatorWrapper(
                tokenizer=self.diff_tokenizer, is_train_val=True
            )
        return self._train_val_collator

    @property
    def test_collator(self):
        if self._test_collator is None:
            self._test_collator = DataCollatorWrapper(
                tokenizer=self.diff_tokenizer, is_train_val=False
            )
        return self._test_collator

    def prepare_data(self) -> None:
        """Prepares the dataset by processing and tokenizing data for training, validation, and testing splits.
        Note: Do not use it to assign state (self.x = y).
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
                    fn_kwargs={
                        "diff_tokenizer": self.diff_tokenizer,
                        "msg_tokenizer": self.msg_tokenizer,
                        "diff_max_len": self.hparams.diff_max_len,
                        "msg_max_len": self.hparams.msg_max_len,
                    },
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset.column_names,
                    num_proc=max(1, multiprocessing.cpu_count() - 1),
                )
                dataset.save_to_disk(tokenized_path)

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
            dataset = load_from_disk(self.tokenized_data_dir / split)
            # dataset.set_format("torch")
            return dataset

        # load datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = _load_dataset("train")
            self.data_val = _load_dataset("validation")
            self.data_test = _load_dataset("test")

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
            collate_fn=self.test_collator,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )


class DataCollatorWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, is_train_val: bool):
        self.collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        self.is_train_val = is_train_val

    def __call__(self, examples):
        batch = self.collator(examples)
        return Batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # the decoder_input_ids and decoder_attention_mask are automatically calculated from the labels by the
            # HuggingFace model, so we can have them as None here.
            # See https://huggingface.co/docs/transformers/glossary#decoder-input-ids
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=batch["labels"],
        )


def tokenize_function(examples, diff_tokenizer, msg_tokenizer, diff_max_len, msg_max_len):
    # See https://huggingface.co/docs/transformers/en/tasks/summarization#preprocess
    inputs = diff_tokenizer(text=examples["diff"], max_length=diff_max_len, truncation=True)
    # If u confused about text_target argument, see https://stackoverflow.com/a/76167575/7121776
    labels = msg_tokenizer(text_target=examples["msg"], max_length=msg_max_len, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs
