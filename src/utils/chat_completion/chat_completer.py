from pathlib import Path
from typing import Any

from dataclasses import dataclass
from datasets import Dataset


class LLMChatCompleter:
    def count_tokens(self, message: str | list[dict[str, str]]) -> int:
        raise NotImplementedError()

    def encode(self, message: str) -> list[int]:
        raise NotImplementedError()

    def decode(self, message: list[int]) -> str:
        raise NotImplementedError()

    def complete_chat(self, messages: list[dict[str, str]]) -> "LLMChatCompleterResponse":
        raise NotImplementedError()

    def prepare_batch(self, dataset: Dataset, output_dir: Path) -> None:
        raise NotImplementedError()

    def submit_batch(self, working_dir: Path, output_dataset_path: Path) -> None:
        raise NotImplementedError()

    def estimate_max_costs(self, dataset_size: int) -> dict[str, Any]:
        return {}


@dataclass
class LLMChatCompleterResponse:
    content: str
    prompt_token_count: int
    response_token_count: int
