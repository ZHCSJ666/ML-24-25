from pathlib import Path
from typing import Any, Optional, List, Literal

import datasets
import torch
from datasets import load_from_disk
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig

from data.components.collators import DataCollatorTrain, DataCollatorTest
from data.commit_chronicle.preprocessors import CommitChroniclePreprocessor
from data.components.tokenization import load_tokenizers
from data.types import SingleExample


class CommitChronicleDataModule(LightningDataModule):
    """`LightningDataModule` for the Commit Chronicle dataset.

    This is the dataset for commit message generation (and/or completion), introduced in the paper
    "From Commit Message Generation to History-Aware Commit Message Completion", ASE 2023.

    Read the docs:
        https://huggingface.co/datasets/JetBrains-Research/commit-chronicle
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/datasets/commit-chronicle",
        languages: List[str] = ["Go"],
        diff_tokenizer_name_or_path: str = "Salesforce/codet5-base",
        msg_tokenizer_name_or_path: str = "Salesforce/codet5-base",
        model_configuration: str = "encoder",
        encoder_context_max_len: int = 512,
        decoder_context_max_len: int = 512,
        shift_labels: bool = True,
        # input configuration
        encoder_input_type: Literal["diff", "history"] = "diff",
        train_with_history: bool = False,
        generate_with_history: bool = False,
        context_ratio: float = 0.0,
        line_sep: str = "\n",
        # preprocessing
        add_history_to_inputs: bool = False,
        use_cache: bool = True,
        # data loader stuff
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CommitDataModule`.

        Args:
            data_dir: Directory with data, should contain files `train.jsonl`, `val.jsonl`, `test.jsonl`.
            diff_tokenizer_name_or_path:
            msg_tokenizer_name_or_path:
            encoder_context_max_len:
            decoder_context_max_len:
            add_history_to_inputs: True to save history for each input example,
                False to load history in RAM and build inputs on the fly.
            line_sep: Newline separator used in data (should be the same for diffs and messages).
            use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.
            languages:
            batch_size: The batch size. Defaults to `16`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
            shift_labels: Should be True most times, except when using a decoder-only model
            encoder_input_type: What type of input will be passed to encoder. Currently, `history` and `diff` are supported.
            train_with_history: `True` to concatenate commit message history with current commit message in decoder
                context during training, `False` otherwise (ignored when `encoder_input_type` is `history`).
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.diff_tokenizer, self.msg_tokenizer = load_tokenizers(
            msg_tokenizer_name_or_path, diff_tokenizer_name_or_path
        )
        self.processor = CommitChroniclePreprocessor(
            diff_tokenizer=self.diff_tokenizer,
            msg_tokenizer=self.msg_tokenizer,
            diff_line_sep=line_sep,
            diff_max_len=encoder_context_max_len,
            add_history_to_inputs=add_history_to_inputs,
            decoder_context_max_length=decoder_context_max_len,
        )
        self.vocab_size = self.msg_tokenizer.vocab_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_val_collator = None
        self.test_collator = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        for split in ["train", "validation", "test"]:
            self.processor.process(
                data_dir=(Path(self.hparams.data_dir)),
                split=split,
                use_cache=self.hparams.use_cache,
            )

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
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        def create_dataset(split):
            path = self.processor.processed_path_for(Path(self.hparams.data_dir), split)
            dataset = load_from_disk(path)
            return CommitChronicleDataset(dataset)

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = create_dataset("train")
            self.data_val = create_dataset("validation")
            self.data_test = create_dataset("test")

            self.train_val_collator = DataCollatorTrain(
                diff_bos_token_id=self.diff_tokenizer.bos_token_id,  # type: ignore[attr-defined]
                diff_eos_token_id=self.diff_tokenizer.eos_token_id,  # type: ignore[attr-defined]
                diff_pad_token_id=self.diff_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                msg_bos_token_id=self.msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
                msg_eos_token_id=self.msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
                msg_pad_token_id=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                msg_sep_token_id=self.msg_tokenizer.sep_token_id,  # type: ignore[attr-defined]
                encoder_input_type=self.hparams.encoder_input_type,
                encoder_context_max_len=self.hparams.encoder_context_max_len,
                decoder_context_max_len=self.hparams.decoder_context_max_len,
                with_history=self.hparams.train_with_history,
                process_retrieved=False,
                shift_labels=self.hparams.shift_labels,
                testing=False,
                decoder_start_token_id=get_decoder_start_token_id(
                    self.hparams.model_configuration
                ),
            )
            self.test_collator = DataCollatorTest(
                diff_bos_token_id=self.diff_tokenizer.bos_token_id,  # type: ignore[attr-defined]
                diff_eos_token_id=self.diff_tokenizer.eos_token_id,  # type: ignore[attr-defined]
                diff_pad_token_id=self.diff_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                msg_bos_token_id=self.msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
                msg_eos_token_id=self.msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
                msg_pad_token_id=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                msg_sep_token_id=self.msg_tokenizer.sep_token_id,  # type: ignore[attr-defined]
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                encoder_input_type=self.hparams.encoder_input_type,
                encoder_context_max_len=self.hparams.encoder_context_max_len,
                decoder_context_max_len=self.hparams.decoder_context_max_len,
                with_history=self.hparams.generate_with_history,
                context_ratio=self.hparams.context_ratio,
                process_retrieved=False,
                testing=False,
                decoder_start_token_id=get_decoder_start_token_id(
                    self.hparams.model_configuration
                ),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.train_val_collator,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.train_val_collator,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.test_collator,
        )


def get_decoder_start_token_id(model_cfg: str) -> Optional[int]:
    if model_cfg == "encoder_decoder":
        return None
    elif model_cfg == "decoder":
        return None
    elif model_cfg == "encoder":
        return None
    # assumes seq2seq
    config = AutoConfig.from_pretrained(model_cfg)
    return config.decoder_start_token_id


class CommitChronicleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> SingleExample:
        row = self.dataset[index]
        return SingleExample(
            diff_input_ids=row["diff_input_ids"],
            msg_input_ids=row["msg_input_ids"],
            history_input_ids=row.get("history_input_ids"),
            pos_in_file=row.get("pos_in_file"),
        )


if __name__ == "__main__":
    _ = CommitChronicleDataModule()
