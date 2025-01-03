import hashlib
import random
from typing import Optional

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


class TextLoggingMixin:
    def log_text(self, prefix, results: list[dict[str, str]], num_results: int = 4) -> None:
        """Log generated git commit message results.

        This method only supports TensorBoard and Wandb at the moment.
        """
        random.shuffle(results)
        self.log_text_tensorboard(prefix, results, num_results)
        self.log_text_wandb(prefix, results, num_results)

    def log_text_tensorboard(self, prefix, results: list[dict[str, str]], num_results) -> None:
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

    def log_text_wandb(self, prefix, results: list[dict[str, str]], num_results) -> None:
        wandb_logger: Optional[WandbLogger] = None
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        if wandb_logger is None:
            return

        columns = list(results[0].keys())
        data = [list(result.values()) for result in results[:num_results]]
        wandb_logger.log_text(prefix + "text", columns=columns, data=data, step=self.global_step)


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
