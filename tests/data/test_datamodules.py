from pathlib import Path

import rootutils

from src.data.commit_chronicle import CommitChronicleDataModule

ROOT = rootutils.find_root(__file__, ".project-root")


def test_commit_chronicle_datamodule() -> None:
    """Tests `CommitChronicleDataModule` to verify that it can be downloaded correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
    sizes correctly match."""
    data_dir = str(ROOT / "data/datasets/commit-chronicle")

    dm = CommitChronicleDataModule(data_dir=data_dir, batch_size=2)
    dm.prepare_data()

    assert not dm.data_test
    assert Path(data_dir).exists()
    assert Path(data_dir, "processed/test/dataset_info.json").exists()

    dm.setup()
    assert dm.data_test and len(dm.data_test) and dm.data_test[0]
    print(dm.data_test[0])
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
