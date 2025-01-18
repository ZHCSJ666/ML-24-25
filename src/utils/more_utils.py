import hashlib
import json
from pathlib import Path
from typing import Optional

import wandb
from datasets import Dataset, load_dataset
from lightning.pytorch.loggers import TensorBoardLogger


class TextLoggingMixin:
    _wandb_tables: dict[str, wandb.Table]

    def log_text(
        self, prefix, results: list[dict[str, str]], num_results: Optional[int] = None
    ) -> None:
        """Log generated git commit message results.

        This method only supports TensorBoard and Wandb at the moment.
        """
        self._log_text_tensorboard(prefix, results, num_results)
        self._log_text_wandb(prefix, results, num_results)

    def _log_text_tensorboard(
        self, prefix, results: list[dict[str, str]], num_results: Optional[int]
    ) -> None:
        tb_logger: Optional[TensorBoardLogger] = None
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break
        if tb_logger is None:
            return

        writer = tb_logger.experiment
        for result in results[:num_results]:
            for key, value in result.items():
                writer.add_text(prefix + key, value, self.global_step)

    def _log_text_wandb(
        self,
        prefix,
        results: list[dict[str, str]],
        num_results: Optional[int],
        use_step: bool = True,
    ) -> None:
        if num_results:
            results = results[:num_results]
        if self._is_wandb_initialized():
            columns = ["step" if use_step else "epoch"] + list(results[0].keys())
            step_or_epoch = [self.global_step if use_step else self.current_epoch]
            # inspired by https://github.com/wandb/wandb/issues/2981#issuecomment-2495140257
            saved_table = self._get_wandb_table(prefix) or wandb.Table(columns=columns)
            table = wandb.Table(columns=columns, data=saved_table.data)  #
            for result in results:
                row_data = step_or_epoch + list(result.values())
                table.add_data(*row_data)
            wandb.log({prefix: table}, commit=False)
            self._set_wandb_table(prefix, table)

    @staticmethod
    def _is_wandb_initialized() -> bool:
        return wandb.run is not None

    def _get_wandb_table(self, prefix: str) -> Optional[wandb.Table]:
        if not hasattr(self, "_wandb_tables"):
            self._wandb_tables = {}
        return self._wandb_tables.get(prefix)

    def _set_wandb_table(self, prefix: str, table: wandb.Table) -> None:
        if not hasattr(self, "_wandb_tables"):
            self._wandb_tables = {}
        self._wandb_tables[prefix] = table


def hash_dict(input_dict):
    """
    Generates a hash for a dictionary.

    Args:
        input_dict (dict): The dictionary to hash.

    Returns:
        str: A hexadecimal hash string.
    """
    # Ensure the dictionary is sorted to maintain consistency
    dict_as_tuple = tuple(sorted(input_dict.items()))
    # Convert the tuple to a string and encode to bytes
    dict_as_bytes = str(dict_as_tuple).encode("utf-8")
    # Hash the bytes using SHA256
    hash_object = hashlib.sha256(dict_as_bytes)
    return hash_object.hexdigest()


def load_jsonl_as_dataset(path: Path, split: str = "train") -> Dataset:
    dataset = load_dataset("json", data_files={split: str(path)})[split]
    return dataset


def get_last_checkpoint(checkpoint_file: Path) -> int:
    """Retrieve the last processed index from the checkpoint file."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            try:
                return int(f.read().strip())  # Read last index
            except ValueError:
                return -1  # Default to start from the beginning
    return -1  # If no checkpoint file, start from scratch


def append_jsonl(data: dict, filename: Path) -> None:
    """Append a JSON object as a new line in a JSONL file."""
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


def save_checkpoint(checkpoint_file: Path, index: int) -> None:
    """Save the last processed index to the checkpoint file."""
    with open(checkpoint_file, "w") as f:
        f.write(str(index))
