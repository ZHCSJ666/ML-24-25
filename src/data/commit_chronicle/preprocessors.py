import linecache
import logging
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import polars as pl
from datasets import load_dataset, load_from_disk, Dataset
from datasets.formatting.formatting import LazyRow
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class CommitChroniclePreprocessor:
    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        diff_line_sep: str = "\n",
        diff_max_len: int = 512,
        languages: Sequence[str] = ("Go",),
        add_history_to_inputs: bool = False,
        decoder_context_max_length: Optional[int] = None,
        cpu_count: Optional[int] = None,
    ) -> None:
        """

        Args:
            diff_tokenizer:
            msg_tokenizer:
            diff_line_sep:  Line separator in diffs.
            diff_max_len:
            languages:
            add_history_to_inputs: True to add history inputs to each example in a processed file.
            decoder_context_max_length: Should be provided when add_history_to_inputs is True.
            cpu_count:
        """
        super().__init__()

        assert not add_history_to_inputs or (
            add_history_to_inputs and decoder_context_max_length is not None
        ), "You have to define max context length to aggregate history in inputs."

        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer
        self._num_commits: Dict[int, int] = defaultdict(int)
        self.diff_line_sep = diff_line_sep
        self.diff_max_len = diff_max_len
        self.languages = languages
        self.cpu_count = cpu_count or mp.cpu_count()
        self.add_history_to_inputs = add_history_to_inputs
        self.decoder_context_max_length = decoder_context_max_length

    def processed_path_for(self, data_dir: Path, split: str) -> Path:
        processed_path = data_dir / "processed" / split
        processed_history_path = data_dir / "processed+history" / split
        return processed_history_path if self.add_history_to_inputs else processed_path

    def process(
        self,
        data_dir: Path,
        split: str,
        use_cache: bool,
    ) -> Path:
        """
        Main processing logic.

        1. Iterate over input file in chunks, process and tokenize messages and diffs, save to separate file.
        2. Aggregate history from processed file, save to separate file.
        3. If add_history_to_inputs is True, iterate through input file again and aggregate history inputs for each example.
        4. If processing train, shuffle processed file and save to yet another separate file.

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
                load_dataset(
                    "JetBrains-Research/commit-chronicle", "default", split=split
                )
                .filter(
                    lambda example: example["language"] in self.languages,
                    num_proc=self.cpu_count,
                )
                .map(self._process_example, num_proc=self.cpu_count)
                .save_to_disk(processed_path)
            )

        history_path = data_dir / "history" / split
        if use_cache and history_path.exists():
            logging.info(f"{history_path} found, won't rewrite")
        else:
            logging.info("Processing history")
            self._process_history(
                input_path=processed_path,
                output_path=history_path,
            )

        processed_history_path = data_dir / "processed+history" / split
        if self.add_history_to_inputs:
            if use_cache and processed_history_path.exists():
                logging.info(f"{processed_history_path} found, won't rewrite")
            else:
                self._add_history_to_inputs(
                    input_path=processed_path,
                    history_path=history_path,
                    output_path=processed_history_path,
                    decoder_context_max_length=self.decoder_context_max_length,
                )

        return processed_history_path if self.add_history_to_inputs else processed_path

    def _process_example(self, example: LazyRow) -> Dict[str, Any]:
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
            "pos_in_history": self._get_pos_in_history([example["author"]])[0],
        }

    def _preprocess_diff(self, diff: str) -> str:
        """Return given diff without any changes."""
        return diff

    def _preprocess_mods(self, mods: List[Dict[str, str]]) -> str:
        """
        Transforms a list of all files modifications made in a commit into a single string representation.

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
                file_diff = (
                    f"copy from {mod['old_path']}{line_sep}copy to {mod['new_path']}"
                )
            else:
                file_diff = f"{mod['new_path']}"
            diff += file_diff + line_sep + self._preprocess_diff(mod["diff"])

        return diff

    def _preprocess_message(self, message: str) -> str:
        """Returns given message without any changes."""
        return message

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        """Tokenizes diffs via transformers' tokenizer.

        Diffs are truncated to save memory. Special tokens are added later, during batch construction, so 2 extra tokens
        from max_length are reserved for BOS and EOS.
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
        return self._msg_tokenizer(
            messages, truncation=False, padding=False, add_special_tokens=False  # type: ignore[operator]
        ).input_ids

    def _process_history(self, input_path: Path, output_path: Path) -> None:
        """
        Aggregates commit message history for each author in a given file.

        Input file should be in JSONL format and contain keys "author" and "msg_input_ids".
        Also, messages from each author are expected to be in chronological order
        (it won't break anything, but logically the result would be incorrect).

        Output is a JSON file with authors ids as keys and lists of their messages as values.

        Args:
            input_path: Path to file to read data from.
            output_path: Path to file to save history to.
        """
        df = (
            load_from_disk(input_path)
            .map(
                lambda row: {
                    "author": row["author"],
                    "msg_input_ids": row["msg_input_ids"],
                },
                num_proc=self.cpu_count,
            )
            # https://github.com/huggingface/datasets/issues/3644#issuecomment-1997484590
            .to_polars()
            .group_by("author")
            .agg(pl.col("msg_input_ids"))
        )
        Dataset.from_polars(df).save_to_disk(output_path)

    def _add_history_to_inputs(
        self,
        input_path: Path,
        history_path: Path,
        output_path: Path,
        decoder_context_max_length: int,
    ) -> None:
        """Adds commit message history to each example in the input file and saves the results to the output file.

        This approach uses more disk space but enables working with the dataset in a fully iterable fashion
        without loading the history into RAM. To prevent excessive disk usage, the messages from history are added only
        until the maximum decoder context length is achieved.

        Args:
            input_path: Path to file to read data from.
            history_path: Path to file to read history from.
            output_path: Path to file to save data with history inputs to.
            decoder_context_max_length: Maximum allowed number of tokens in decoder context.
        """

        def process_row(row: LazyRow):
            all_author_history: List[List[int]] = history[int(row["author"])][
                : row["pos_in_history"]
            ]
            relevant_author_history: List[List[int]] = []
            cur_len = (
                len(row["msg_input_ids"]) + 2
            )  # +2 to account for BOS and EOS tokens
            for history_msg in all_author_history[::-1]:
                if cur_len + len(history_msg) + 1 > decoder_context_max_length:
                    break
                relevant_author_history.append(history_msg)
                cur_len += len(history_msg) + 1
            row["history_input_ids"] = relevant_author_history[::-1]
            return row

        history = {
            int(entry["author"]): entry["msg_input_ids"]
            for entry in load_from_disk(history_path)
        }

        (
            load_from_disk(input_path)
            .map(process_row, num_proc=self.cpu_count)
            .save_to_disk(output_path)
        )

    def _get_pos_in_history(self, authors: List[int]) -> List[int]:
        """Builds correct position in history for each commit when iterating over input data
        in chunks.

        Args:
            authors: A list of authors for commits from the current chunk.

        Returns:
            A list of positions in the corresponding author's history for each commit from the chunk.
        """
        positions_in_history = []
        for author in authors:
            self._num_commits[author] += 1
            positions_in_history.append(self._num_commits[author] - 1)
        return positions_in_history

    def _shuffle(self, input_path: str, output_path: str) -> None:
        """Shuffles a file.

        To support moderately large files, it works by shuffling a list of line idxs
        and then utilizing `linecache` to write specific lines in a new order.

        Args:
            input_path: Path to input file.
            output_path: Path to output file.
        """
        random.seed(42)
        logging.info("Calculating number of lines")
        with open(input_path) as f:
            num_lines = sum(1 for _ in f)
        logging.info("Shuffling line idxs")
        idxs = [
            i + 1 for i in range(num_lines)
        ]  # start rows idxs with 1, since linecache starts with 1
        random.shuffle(idxs)
        with open(output_path, "w") as f:
            for i in tqdm(idxs, f"Writing shuffled lines for {input_path}..."):
                f.write(linecache.getline(input_path, i))
