import multiprocessing
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets import load_from_disk
from datasets.formatting.formatting import LazyBatch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling

from src.data.commit_chronicle.preprocessor import CommitChroniclePreprocessor
from utils.more_utils import hash_dict


class CommitChronicleCLMDataModule(LightningDataModule):
    """Commit Chronicle dataset `LightningDataModule` for Causal Language Modeling (CLM).

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
        truncate_training_data: bool = False,
        line_sep: str = "\n",
        use_cache: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize a `CommitChronicleCLMDataModule`.

        Args:
            tokenizer: Tokenizer used to tokenize both diffs and commit messages.
            data_dir: Directory to save preprocessed data.
            languages: The languages to be included in the dataset.
            change_types: The change types to be included in the dataset.
                Each element should be one of 'ADD', 'DELETE', 'RENAME', 'COPY' or 'MODIFY'
            input_max_len: Maximum length for concatenated input git commit diff and messages.
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

        self.data_collator = DataCollatorWrapper(tokenizer)
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
            "tokenizer": self.tokenizer.name_or_path,
            "input_max_len": self.hparams.input_max_len,
            "truncate_data": self.hparams.truncate_training_data,
        }
        return Path(self.hparams.data_dir) / f"clm-{hash_dict(config)}"

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
                    batch_tokenize_function,
                    fn_kwargs={
                        "tokenizer": self.tokenizer,
                        "max_length": self.hparams.input_max_len,
                        "truncate_data": split == "test" or self.hparams.truncate_training_data,
                        # for test split, we want to simulate real world scenario where only have git diffs
                        # without ground truth commit message. In this case, input will be of format `git diff <|sep|>`
                        "add_msg_to_truncated": split != "test",
                    },
                    remove_columns=dataset.column_names,
                    batched=True,
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
            return load_from_disk(self.tokenized_data_dir / split)

        # load and split datasets only if not loaded already
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
            collate_fn=self.data_collator,
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
            collate_fn=self.data_collator,
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
            collate_fn=self.data_collator,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
        )


class DataCollatorWrapper:
    def __init__(self, tokenizer):
        self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, batch):
        targets = [example.pop("target", None) for example in batch]
        outputs = self.collator(batch)
        outputs["target"] = targets
        return outputs


def batch_tokenize_function(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    truncate_data: bool,
    add_msg_to_truncated: bool,
):
    """

    Args:
        examples:
        tokenizer:
        max_length:
        truncate_data:
        add_msg_to_truncated:

    Returns:

    Helpful discussions:
    - https://discuss.huggingface.co/t/how-does-gpt-decide-to-stop-generating-sentences-without-eos-token/41623/4
    """
    if truncate_data:
        diff_input_ids = tokenizer(
            text=examples["diff"],
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=sys.maxsize,  # to suppress a warning
        )["input_ids"]

        if add_msg_to_truncated:
            msg_input_ids = tokenizer(
                text=examples["msg"],
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                max_length=sys.maxsize,  # to suppress a warning
            )["input_ids"]
        else:
            msg_input_ids = [[] for _ in examples["msg"]]

        input_ids = [
            concat_diff_and_msg(diff, msg, max_length, tokenizer.sep_token_id)
            for diff, msg in zip(diff_input_ids, msg_input_ids)
        ]

        return {"input_ids": input_ids, "target": examples["msg"]}

    # outputs = tokenizer(
    #     text=f"{examples['diff']} {tokenizer.sep_token} {examples['msg']}",
    #     max_length=max_length,
    #     truncation=True,
    #     return_overflowing_tokens=True,
    #     return_length=True,
    #     add_special_tokens=False,
    #     return_attention_mask=False,
    # )
    # input_batch = []
    # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
    #     if length == max_length:
    #         input_batch.append(input_ids)
    # return {"input_ids": input_batch}

    assert tokenizer.sep_token is not None

    input_ids_batch = tokenizer(
        text=[
            f"{examples['diff'][idx]} {tokenizer.sep_token} {examples['msg'][idx]}"
            for idx in range(len(examples["diff"]))
        ],
        truncation=False,
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=sys.maxsize,  # to suppress a warning
    )["input_ids"]

    # Add eos_token_id between samples
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer does not have an eos_token_id.")

    # flatten the tokens and insert eos_token_id
    concatenated_ids = []
    for input_ids in input_ids_batch:
        concatenated_ids.extend(input_ids + [eos_token_id])

    # Remove the trailing eos_token_id
    if concatenated_ids[-1] == eos_token_id:
        concatenated_ids = concatenated_ids[:-1]

    # Chunk the concatenated sequence into `max_length` pieces
    chunks = [
        concatenated_ids[i : i + max_length] for i in range(0, len(concatenated_ids), max_length)
    ]

    return {"input_ids": chunks}


def concat_diff_and_msg(diff_input_ids, msg_input_ids, max_length, sep_token_id):
    assert sep_token_id is not None

    num_special_characters = 1
    # number of msg tokens should not be more than half the total input size
    max_msg_len = (max_length - num_special_characters) // 2
    msg_input_ids = msg_input_ids[:max_msg_len]
    diff_input_ids = diff_input_ids[
        : max_length - num_special_characters - min(len(msg_input_ids), max_msg_len)
    ]

    return diff_input_ids + [sep_token_id] + msg_input_ids
