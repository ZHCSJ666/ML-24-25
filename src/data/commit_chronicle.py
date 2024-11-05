from pathlib import Path
from typing import Any, Optional, Tuple, List

from datasets import load_dataset, load_from_disk, DatasetDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


# noinspection PyDefaultArgument
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
        data_dir: str = "data/commit-chronicle",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        languages: List[str] = ["C#"],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CommitDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.dataset: Optional[DatasetDict] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        data_dir = Path(self.hparams.data_dir)
        if not data_dir.exists():
            dataset = load_dataset("JetBrains-Research/commit-chronicle", "default")
            dataset = dataset.filter(
                lambda example: example["language"] in self.hparams.languages
            )
            dataset.save_to_disk(data_dir)

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

        # load and split datasets only if not loaded already
        if not self.dataset:
            self.dataset = load_from_disk(self.hparams.data_dir)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = CommitChronicleDataModule()
