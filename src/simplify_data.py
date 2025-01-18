import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import hydra
import rootutils
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utils.chat_completion import LLMChatCompleter
from src.utils.more_utils import (
    hash_dict,
    load_jsonl_as_dataset,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@dataclass
class MainDictConfig:
    split: str
    languages: list[str]
    change_types: list[str]
    diff_line_sep: str

    chat_completer: LLMChatCompleter

    system_content: str
    user_content_template: str
    max_prompt_token_count: int
    max_msg_token_count: int

    root_dir: str
    output_dir: str

    seed: int
    mode: Literal["prompt-testing", "eager", "batch"]
    prompt_testing_example_index: int

    debug_run: bool
    huggingface_repo: str | None = None


def run_completion(example, completer: LLMChatCompleter):
    response = completer.complete_chat(
        [
            {
                "role": "system",
                "content": example["system_content"],
            },
            {
                "role": "user",
                "content": example["user_content"],
            },
        ]
    )
    return {"message": response.content.strip()}


def ensure_paths(cfg: DictConfig, split_names: list[str]):
    cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # create directory and config file for filtered data
    config = {
        "languages": sorted(cfg["languages"]),
        "change_types": sorted(cfg["change_types"]),
        "diff_line_sep": cfg["diff_line_sep"],
    }
    hashed_config = hash_dict(config)
    filtered_data_paths = [
        output_dir / "filtered" / hashed_config / f"{split}.jsonl" for split in split_names
    ]

    filtered_data_meta = output_dir / "filtered" / hashed_config / "config.json"
    filtered_data_meta.parent.mkdir(exist_ok=True, parents=True)
    with open(filtered_data_meta, "w") as f:
        json.dump(config, f, indent=4)

    # create directory and config file for intermediate files
    config = config | {
        "system_content": cfg["system_content"],
        "user_content_template": cfg["user_content_template"],
        "max_prompt_token_count": cfg["max_prompt_token_count"],
        "chat_completer": cfg["chat_completer"],
    }
    hashed_config = hash_dict(config)
    working_dirs = [output_dir / hashed_config / split for split in split_names]
    working_dir_config = output_dir / hashed_config / "config.json"
    working_dir_config.parent.mkdir(exist_ok=True, parents=True)
    with open(working_dir_config, "w") as f:
        json.dump(config, f, indent=4)

    # final output path
    output_data_paths = [
        output_dir / hashed_config / f"{split}-final.jsonl" for split in split_names
    ]
    combined_final_output_path = output_dir / hashed_config / "final"
    return (
        filtered_data_paths,
        working_dirs,
        output_data_paths,
        combined_final_output_path,
    )


def filter_example(example, languages: set[str], change_types: set[str]):
    return example["language"] in languages and (
        not change_types or all(mod["change_type"] in change_types for mod in example["mods"])
    )


def validate_user_content_template(message_template: str):
    assert "{diff}" in message_template
    assert "{message}" in message_template


def process_example(
    example,
    completer: LLMChatCompleter,
    system_content: str,
    user_content_template: str,
    max_prompt_token_count: int,
    max_msg_token_count: int,
    return_token_stats: bool = False,
) -> dict[str, Any]:
    validate_user_content_template(user_content_template)

    message = example["message"]
    diff = example["diff"]

    non_diff_msg_token_count = completer.count_tokens(
        [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content_template.format(diff="", message=""),
            },
        ],
    )
    orig_msg_token_count = completer.count_tokens(message)
    msg_token_count = min(orig_msg_token_count, max_msg_token_count)
    max_diff_token_count = max_prompt_token_count - non_diff_msg_token_count - msg_token_count
    assert (
        max_diff_token_count > 0
    ), f"Max diff token count must be more than zero, num_msg_tokens={msg_token_count}, num_other_tokens={non_diff_msg_token_count}"

    # truncate diff to `max_diff_token_count`
    diff_tokens = completer.encode(diff)
    orig_diff_token_count = len(diff_tokens)
    diff_tokens = diff_tokens[:max_diff_token_count]
    diff = completer.decode(diff_tokens)

    output = {
        "system_content": system_content,
        "user_content": user_content_template.format(diff=diff, message=message),
    }
    if return_token_stats:
        output["num_diff_tokens"] = orig_diff_token_count
        output["num_msg_tokens"] = orig_msg_token_count
        output["num_other_tokens"] = non_diff_msg_token_count
        output["num_total_tokens"] = (
            orig_diff_token_count + orig_msg_token_count + non_diff_msg_token_count
        )
    return output


def preprocess_mods(mods: list[dict[str, str]], line_sep: str) -> str:
    """Transforms a list of all files modifications made in a commit into a single string
    representation.

    Specifically, adds a header to each file diff (https://git-scm.com/docs/git-diff#_generating_patch_text_with_p)
    and concatenates the results.

    Args:
        mods: A list of files modifications made in a commit.

    Returns:
        A single string representation of all files modifications made in a commit.
    """
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
        diff += file_diff + line_sep + mod["diff"]
    return diff


def simplify_dataset_by_split(
    cfg: MainDictConfig,
    filtered_data_path: Path,
    working_dir: Path,
    output_dataset_path: Path,
    split: str,
):
    completer = cfg.chat_completer

    # filter data and save as jsonl
    if filtered_data_path.exists():
        filtered_dataset = load_jsonl_as_dataset(filtered_data_path)
    else:
        filtered_dataset = load_dataset(
            "JetBrains-Research/commit-chronicle", "default", split=split
        )
        filtered_dataset = filtered_dataset.filter(
            filter_example,
            fn_kwargs={"languages": cfg.languages, "change_types": cfg.change_types},
            num_proc=mp.cpu_count(),
        )
        filtered_dataset = filtered_dataset.map(
            lambda example, diff_line_sep: {
                "diff": preprocess_mods(example["mods"], diff_line_sep)
            },
            fn_kwargs={"diff_line_sep": cfg.diff_line_sep},
            num_proc=mp.cpu_count(),
        )
        filtered_dataset.to_json(filtered_data_path, orient="records", lines=True)

    logger.debug(cfg.user_content_template)
    logger.info("Size of dataset: {Size}", Size=len(filtered_dataset))

    costs = completer.estimate_max_costs(len(filtered_dataset))
    logger.info(f"Max API costs: {costs}")

    if cfg.mode == "prompt-testing":
        idx = (
            random.randint(0, len(filtered_dataset) - 1)
            if cfg.prompt_testing_example_index == -1
            else cfg.prompt_testing_example_index
        )
        dataset = filtered_dataset.select([idx]).map(
            process_example,
            fn_kwargs={
                "system_content": cfg.system_content,
                "user_content_template": cfg.user_content_template,
                "max_prompt_token_count": cfg.max_prompt_token_count,
                "max_msg_token_count": cfg.max_msg_token_count,
                "completer": completer,
                "return_token_stats": True,
            },
        )
        for example in dataset:
            response = completer.complete_chat(
                [
                    {
                        "role": "system",
                        "content": example["system_content"],
                    },
                    {
                        "role": "user",
                        "content": example["user_content"],
                    },
                ]
            )
            logger.debug(f"Diff: {example['diff']}")
            logger.debug(f"Message (Simplified): {response.content.strip()}")
            logger.debug(f"Message (Original): {example['message']}")
            logger.debug(
                f"Original token stats: num_msg_tokens={example['num_msg_tokens']}, num_diff_tokens={example['num_diff_tokens']}, num_other_tokens={example['num_other_tokens']}, num_total_tokens={example['num_total_tokens']}"
            )
            logger.debug(f"{response.prompt_token_count} prompt tokens counted by the LLM.")
            return

    if cfg.debug_run:
        filtered_dataset = filtered_dataset.select(range(5))

    if cfg.mode == "eager":
        logger.info("Eager mode processing...")

        # create LLM completer request dataset
        api_request_path = working_dir / "requests.jsonl"
        if api_request_path.exists():
            dataset = load_jsonl_as_dataset(api_request_path)
        else:
            dataset = filtered_dataset.map(
                process_example,
                fn_kwargs={
                    "system_content": cfg.system_content,
                    "user_content_template": cfg.user_content_template,
                    "max_prompt_token_count": cfg.max_prompt_token_count,
                    "max_msg_token_count": cfg.max_msg_token_count,
                    "completer": completer,
                },
                num_proc=mp.cpu_count(),
                remove_columns=filtered_dataset.column_names,
            )
            dataset.to_json(api_request_path, orient="records", lines=True)

        # invoke LLM completer to generate results
        api_result_path = working_dir / "results.jsonl"
        if not api_result_path.exists():
            dataset = dataset.map(
                run_completion,
                fn_kwargs={"completer": completer},
                num_proc=mp.cpu_count(),
                remove_columns=dataset.column_names,
            )
            dataset.to_json(api_result_path, orient="records", lines=True)
        else:
            dataset = load_jsonl_as_dataset(api_result_path)

        # merge api results with filtered data to create simplified dataset
        output_dataset = filtered_dataset.remove_columns(["diff", "message"]).add_column(
            "message", dataset["message"]
        )
        output_dataset.to_json(output_dataset_path, orient="records", lines=True)

        # checkpoint_path = split_working_dir / "checkpoint.txt"
        # last_index = get_last_checkpoint(checkpoint_path)
        # logger.info(f"Starting from index: {last_index + 1}")
        #
        # for i, example in tqdm(enumerate(dataset)):
        #     if i <= last_index:
        #         continue
        #     response = completer.complete_chat(
        #         [
        #             {
        #                 "role": "system",
        #                 "content": example["system_content"],
        #             },
        #             {
        #                 "role": "user",
        #                 "content": example["user_content"],
        #             },
        #         ]
        #     )
        #
        #     append_jsonl({"message": response.content.strip()}, final_dataset_path)
        #     save_checkpoint(checkpoint_path, i)
    else:
        dataset = filtered_dataset.map(
            process_example,
            fn_kwargs={
                "system_content": cfg.system_content,
                "user_content_template": cfg.user_content_template,
                "max_prompt_token_count": cfg.max_prompt_token_count,
                "max_msg_token_count": cfg.max_msg_token_count,
                "completer": completer,
            },
            num_proc=mp.cpu_count(),
            remove_columns=filtered_dataset.column_names,
        )
        completer.prepare_batch(dataset, working_dir)
        if not cfg.debug_run:
            api_result_path = completer.submit_batch(working_dir)
            dataset = load_jsonl_as_dataset(api_result_path)

            # merge api results with filtered data to create simplified dataset
            output_dataset = filtered_dataset.remove_columns(["diff", "message"]).add_column(
                "message", dataset["message"]
            )
            output_dataset.to_json(output_dataset_path, orient="records", lines=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="simplify-data.yaml")
def main(config: DictConfig):
    cfg: MainDictConfig = hydra.utils.instantiate(config, _convert_="object")

    # setup & validation
    validate_user_content_template(cfg.user_content_template)

    if cfg.seed is not None:
        random.seed(cfg.seed)

    if cfg.debug_run and cfg.mode != "prompt-testing":
        logger.warning("Doing a debug run...")

    if cfg.mode == "prompt-testing" and not cfg.split:
        logger.warning("Doing prompt-testing but 'split' is not set. Using test split.")
        cfg.split = "test"

    split_names = ["train", "validation", "test"]
    (
        split_filtered_data_paths,
        split_working_dirs,
        split_final_dataset_paths,
        combined_final_output_path,
    ) = ensure_paths(config, split_names)

    for split_name, filtered_data_path, working_dir, output_dataset_path in zip(
        split_names, split_filtered_data_paths, split_working_dirs, split_final_dataset_paths
    ):
        if cfg.split is None or split_name == cfg.split:
            simplify_dataset_by_split(
                cfg, filtered_data_path, working_dir, output_dataset_path, split_name
            )

    if all(path.exists() for path in split_final_dataset_paths) and cfg.mode != "prompt-testing":
        dataset_dict = {
            split_name: load_jsonl_as_dataset(output_dataset_path, split_name)
            for split_name, output_dataset_path in zip(split_names, split_final_dataset_paths)
        }
        dataset_hf = DatasetDict(dataset_dict)
        if cfg.debug_run:
            dataset_hf.save_to_disk(combined_final_output_path)
            logger.success(f"Debug run dataset saved to '{combined_final_output_path}'")
        else:
            login()  # This will prompt you to enter your access token
            dataset_hf.push_to_hub(cfg.huggingface_repo, private=False)
            logger.success(f"Dataset successfully pushed to '{cfg.huggingface_repo}'")


if __name__ == "__main__":
    main()
