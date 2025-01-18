import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import hydra
import rootutils
from datasets import Dataset, DatasetDict, load_dataset
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

    dry_run: bool
    huggingface_repo: str | None = None


@hydra.main(version_base="1.3", config_path="../configs", config_name="simplify-data.yaml")
def main(config: DictConfig):
    cfg: MainDictConfig = hydra.utils.instantiate(config, _convert_="object")

    completer = cfg.chat_completer

    # setup & validation
    validate_user_content_template(cfg.user_content_template)

    if cfg.seed is not None:
        random.seed(cfg.seed)

    filtered_data_path, working_dir, output_dataset_path = ensure_paths(config)

    # filter data and save as jsonl
    if filtered_data_path.exists():
        dataset = load_jsonl_as_dataset(filtered_data_path)
    else:
        dataset = load_dataset("JetBrains-Research/commit-chronicle", "default", split=cfg.split)
        dataset = dataset.filter(
            filter_example,
            fn_kwargs={"languages": cfg.languages, "change_types": cfg.change_types},
            num_proc=mp.cpu_count(),
        )
        dataset = dataset.map(
            lambda example, diff_line_sep: {
                "diff": preprocess_mods(example["mods"], diff_line_sep)
            },
            fn_kwargs={"diff_line_sep": cfg.diff_line_sep},
            num_proc=mp.cpu_count(),
            remove_columns=[name for name in dataset.column_names if name != "message"],
        )
        dataset.to_json(filtered_data_path, orient="records", lines=True)

    logger.debug(cfg.user_content_template)
    logger.info("Size of dataset: {Size}", Size=len(dataset))

    costs = completer.estimate_max_costs(len(dataset))
    logger.info(f"Max API costs: {costs}")

    if cfg.mode == "prompt-testing":
        idx = (
            random.randint(0, len(dataset) - 1)
            if cfg.prompt_testing_example_index == -1
            else cfg.prompt_testing_example_index
        )
        dataset = dataset.select([idx]).map(
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
            logger.debug(f"Simplified: {response.content.strip()}")
            logger.debug(f"Original: {example['message']}")
            logger.debug(
                f"Original token stats: num_msg_tokens={example['num_msg_tokens']}, num_diff_tokens={example['num_diff_tokens']}, num_other_tokens={example['num_other_tokens']}, num_total_tokens={example['num_total_tokens']}"
            )
            logger.debug(f"{response.prompt_token_count} prompt tokens counted by the LLM.")
    elif cfg.mode == "eager":
        logger.info("Eager mode processing...")

        processed_path = working_dir / "processed.jsonl"
        if processed_path.exists():
            dataset = load_jsonl_as_dataset(processed_path)
        else:
            dataset = dataset.map(
                process_example,
                fn_kwargs={
                    "system_content": cfg.system_content,
                    "user_content_template": cfg.user_content_template,
                    "max_prompt_token_count": cfg.max_prompt_token_count,
                    "max_msg_token_count": cfg.max_msg_token_count,
                    "completer": completer,
                    "return_token_stats": True,
                },
                num_proc=mp.cpu_count(),
            )
            dataset.to_json(processed_path, orient="records", lines=True)

        if not output_dataset_path.exists():
            dataset = dataset.map(
                run_completion,
                fn_kwargs={
                    "completer": completer,
                },
                num_proc=mp.cpu_count(),
                remove_columns=dataset.column_names,
            )
            dataset.to_json(output_dataset_path, orient="records", lines=True)

        # checkpoint_path = working_dir / "checkpoint.txt"
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
        #     append_jsonl({"message": response.content.strip()}, output_dataset_path)
        #     save_checkpoint(checkpoint_path, i)
    else:
        dataset = dataset.map(
            process_example,
            fn_kwargs={
                "system_content": cfg.system_content,
                "user_content_template": cfg.user_content_template,
                "max_prompt_token_count": cfg.max_prompt_token_count,
                "max_msg_token_count": cfg.max_msg_token_count,
                "completer": completer,
            },
            num_proc=mp.cpu_count(),
        )
        completer.prepare_batch(dataset, working_dir)
        if not cfg.dry_run:
            completer.submit_batch(working_dir, output_dataset_path)


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


def ensure_paths(cfg: DictConfig):
    cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    config = {
        "languages": sorted(cfg["languages"]),
        "change_types": sorted(cfg["change_types"]),
        "diff_line_sep": cfg["diff_line_sep"],
    }
    hashed_config = hash_dict(config)
    filtered_data_path = output_dir / f"filtered-{hashed_config}.jsonl"
    with open(output_dir / f"filtered-{hashed_config}.meta.json", "w") as f:
        json.dump(config, f)

    config = config | {
        "system_content": cfg["system_content"],
        "user_content_template": cfg["user_content_template"],
        "max_prompt_token_count": cfg["max_prompt_token_count"],
        "chat_completer": cfg["chat_completer"],
    }
    hashed_config = hash_dict(config)
    working_dir = output_dir / f"{hashed_config}"
    working_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / f"{hashed_config}.meta.json", "w") as f:
        json.dump(config, f)
    return filtered_data_path, working_dir, output_dir / f"{hashed_config}.jsonl"


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


def upload_to_huggingface():
    """Upload merged dataset to Hugging Face Hub."""
    splits = ["train", "test", "validation"]
    dataset_dict = {}

    for split in splits:
        merged_jsonl = MERGED_DIR / f"merged_{split}.jsonl"
        data = load_jsonl(merged_jsonl)
        dataset = Dataset.from_list(data)
        dataset_dict[split] = dataset
        print(f"Loaded {len(data)} examples for {split} split.")

    dataset_hf = DatasetDict(dataset_dict)

    print("\nDataset info:")
    for split, dataset in dataset_hf.items():
        print(f"\n{split} split:")
        print(dataset)

    login()

    dataset_hf.push_to_hub(HUGGINGFACE_REPO, private=False)

    print("\nDataset successfully uploaded to Hugging Face Hub!")


if __name__ == "__main__":
    main()
