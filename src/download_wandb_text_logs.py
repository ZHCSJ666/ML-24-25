import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import rootutils
import wandb
from wandb.errors import CommError

ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)


def download_wandb_tables(
    run_id: str,
    output_dir: Path,
    split_names: Sequence[str] = ("test",),
    api_key: str | None = None,
    project: str = "commit-message-generation",
    entity: str = "iasmlproject2024-university-of-hamburg",
) -> dict[str, Path]:
    api_key = api_key or os.environ.get("WANDB_API_KEY")
    assert api_key is not None, "WANDB_API_KEY must be set"
    api = wandb.Api(api_key=api_key)

    tables: dict[str, Path] = dict()
    for split in split_names:
        try:
            wandb_artifact_file = output_dir / run_id / f"{split}.table.json"
            df_file = output_dir / run_id / f"{split}.csv"
            if not wandb_artifact_file.exists():
                artifact = api.artifact(f"{entity}/{project}/run-{run_id}-{split}:latest")
                artifact.download(output_dir / run_id)
            if not df_file.exists():
                with open(wandb_artifact_file, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data["data"], columns=data["columns"])
                df.to_csv(df_file, index=False)
            tables[split] = df_file
        # thrown when artifact is not found
        except CommError:
            pass
    return tables


def create_merged_test_tables() -> None:
    # These runs can be viewed on Weights & Biases
    experiments = [
        {"run_id": "dal1bftd", "name": "t5-efficient-extra-tiny"},
        {"run_id": "6swi6qm9", "name": "baseline-cmg-codet5-without-history"},
    ]
    df_combined = None
    for experiment in experiments:
        path = download_wandb_tables(experiment["run_id"], ROOT / "logs/wandb_downloads")["test"]
        df = pd.read_csv(path)
        if df_combined is None:
            df_combined = df.drop(columns=["step"]).rename(
                columns={"prediction": experiment["name"]}
            )
        else:
            df_combined[experiment["name"]] = df["prediction"]
    # order columns
    df_combined = df_combined[
        ["input"] + [experiment["name"] for experiment in experiments] + ["target"]
    ]
    df_combined.to_csv(ROOT / "logs/comparisons.csv", index=False)


if __name__ == "__main__":
    create_merged_test_tables()
