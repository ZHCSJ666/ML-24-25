import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset
from datasets.formatting.formatting import LazyRow


class CommitChroniclePreprocessor:
    def __init__(
        self,
        diff_line_sep: str = "\n",
        languages: Optional[Sequence[str]] = ("Go",),
        change_types: Optional[Sequence[str]] = ("ADD",),
        cpu_count: Optional[int] = None,
        huggingface_path: str = "JetBrains-Research/commit-chronicle",
    ) -> None:
        """
        Args:
            diff_line_sep:  Line separator in diffs.
            languages:
            change_types:
            cpu_count:
        """
        super().__init__()

        self.diff_line_sep = diff_line_sep
        self.languages = languages
        self.cpu_count = cpu_count or mp.cpu_count()
        self.change_types = change_types
        self.huggingface_path = huggingface_path

    def __call__(self, output_dir: Path, split: str, use_cache: bool) -> None:
        """Main processing logic.

        Args:
            output_dir: Path to directory to save processed files.
            split: Current dataset split.
            use_cache: True to use already processed files when possible, False otherwise.

        Returns:
            Path to processed dataset
        """
        if use_cache and output_dir.exists():
            logging.info(f"Processed data found at '{output_dir}', won't rewrite")
        else:
            dataset = load_dataset(self.huggingface_path, "default", split=split)

            if self.languages:
                logging.info(f"Filtering by languages: {self.languages}")
                dataset = dataset.filter(
                    lambda example: example["language"] in self.languages,
                    num_proc=self.cpu_count,
                )
            if self.change_types:
                logging.info(f"Filtering by change types: {self.change_types}")
                dataset = dataset.filter(
                    lambda example: all(
                        mod["change_type"] in self.change_types for mod in example["mods"]
                    ),
                    num_proc=self.cpu_count,
                )
            dataset.map(self._process_example, num_proc=self.cpu_count).select_columns(
                ["diff", "msg", "repo"]
            ).save_to_disk(output_dir)

    def _process_example(self, example: LazyRow) -> Dict[str, Any]:
        """Processes a single example."""
        message = self._preprocess_message(example["message"])
        mods = self._preprocess_mods(example["mods"])
        return {
            "diff": mods,
            "msg": message,
            "repo": example["repo"],
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
