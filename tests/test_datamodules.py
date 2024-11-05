from pathlib import Path

import pytest

from src.data.commit_chronicle import CommitChronicleDataModule
import rootutils

ROOT = rootutils.find_root(__file__, ".project-root")

@pytest.mark.parametrize("batch_size", [32, 128])
def test_commit_chronicle_datamodule(batch_size: int) -> None:
    """Tests `CommitChronicleDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = str(ROOT / "data/commit-chronicle")

    dm = CommitChronicleDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.dataset
    assert Path(data_dir).exists()
    assert Path(data_dir, "dataset_dict.json").exists()

    dm.setup()
    assert dm.dataset
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

