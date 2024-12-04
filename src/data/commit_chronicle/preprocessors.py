import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset, load_dataset
from datasets.formatting.formatting import LazyRow
from transformers import PreTrainedTokenizerFast


class CommitChroniclePreprocessor:
    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        diff_line_sep: str = "\n",
        diff_max_len: int = 512,
        languages: Sequence[str] = ("Go",),
        change_types: Sequence[str] = ("ADD"),
        msg_max_len: Optional[int] = None,
        cpu_count: Optional[int] = None,
    ) -> None:
        """
        Args:
            diff_tokenizer:
            msg_tokenizer:
            diff_line_sep:  Line separator in diffs.
            diff_max_len:
            languages:
            msg_max_len: Maximum length for commit messages
            cpu_count:
        """
        super().__init__()

        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer
        self.diff_line_sep = diff_line_sep
        self.diff_max_len = diff_max_len
        self.languages = languages
        self.cpu_count = cpu_count or mp.cpu_count()
        self.msg_max_len = msg_max_len
        self.change_types = change_types

    def processed_path_for(self, data_dir: Path, split: str) -> Path:
        processed_path = data_dir / "processed" / split
        return processed_path

    def process(
        self,
        data_dir: Path,
        split: str,
        use_cache: bool,
    ) -> Path:
        """Main processing logic.

        Args:
            data_dir: Path to directory with processed files.
            split: Current dataset split.
            use_cache: True to use already processed files when possible, False otherwise.

        Returns:
            Path to processed dataset
        """
        processed_path = data_dir / "processed" / split

        if use_cache and processed_path.exists():
            logging.info(f"{processed_path} found, won't rewrite")
        else:
            (
                load_dataset("JetBrains-Research/commit-chronicle", "default", split=split)
                .filter(
                    lambda example: example["language"] in self.languages,
                    num_proc=self.cpu_count,
                )
                .filter(
                    lambda example: all(
                        mod["change_type"] in self.change_types for mod in example["mods"]
                    ),
                    num_proc=self.cpu_count,
                )
                .map(self._process_example, num_proc=self.cpu_count)
                .save_to_disk(processed_path)
            )

        return processed_path

    def _process_example(self, example: LazyRow) -> Dict[str, Any]:
        """Processes a single example."""
        message = self._preprocess_message(example["message"])
        mods = self._preprocess_mods(example["mods"])
        return {
            "author": example["author"],
            "message": message,
            "msg_input_ids": self._tokenize_messages([message])[0],
            "diff_input_ids": self._tokenize_diffs([mods])[0],
            "hash": example["hash"],
            "repo": example["repo"],
            "language": example["language"],
        }

    def _preprocess_diff(self, diff: str) -> str:
        """Return given diff without any changes."""
        return diff

    def _preprocess_mods(self, mods: List[Dict[str, str]]) -> str:
        """Transforms a list of all files modifications made in a commit into a single string
        representation.

        Specifically, adds a header to each file diff (https://git-scm.com/docs/git-diff#_generating_patch_text_with_p)
        and concatenates the results.

        Args:
            mods: A list of files modifications made in a commit.

        Returns:
            A single string representation of all files modifications made in a commit.
        """
        line_sep = self.diff_line_sep
        diff = ""
        for mod in mods:
            if mod["change_type"] == "UNKNOWN":
                continue
            elif mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}"
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}"
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}{line_sep}rename to {mod['new_path']}"
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}{line_sep}copy to {mod['new_path']}"
            else:
                file_diff = f"{mod['new_path']}"
            diff += file_diff + line_sep + self._preprocess_diff(mod["diff"])

        return diff

    def _preprocess_message(self, message: str) -> str:
        """Returns given message without any changes."""
        return message

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        """Tokenizes diffs via transformers' tokenizer.

        Diffs are truncated to save memory. Special tokens are added later, during batch
        construction, so 2 extra tokens from max_length are reserved for BOS and EOS.
        """
        tokenized_input = self._diff_tokenizer(
            diffs,
            truncation=True,
            max_length=self.diff_max_len - 2,
            padding=False,
            add_special_tokens=False,
        ).input_ids  # type: ignore[operator]

        return tokenized_input

    def _tokenize_messages(self, messages: List[str]) -> List[List[int]]:
        """Tokenizes commit messages via transformers' tokenizer.'."""
        return self._msg_tokenizer(
            messages, truncation=False, padding=False, add_special_tokens=False  # type: ignore[operator]
        ).input_ids
