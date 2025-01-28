import re
import subprocess
from pathlib import Path

import hydra
import pandas as pd
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import OmegaConf
from loguru import logger

from data.commit_chronicle.llm_api import map_example_to_request
from src.data.commit_chronicle.seq2seq import DataCollatorWrapper

ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)

from src.data.types import Batch


def load_run(checkpoint_path: Path | None, config_path: Path | None, device: torch.device | str | None = None):
    assert config_path is not None or checkpoint_path is not None, "You must specify either config or checkpoint path"
    if config_path is None:
        config_path = checkpoint_path.parent.parent / ".hydra/config.yaml"
    config = OmegaConf.load(config_path)

    # I renamed some classes after training this checkpoint, hence I'm updating the old names here
    # config.data._target_ = "src.data.CommitChronicleSeq2SeqDataModule"
    # config.model._target_ = "src.models.Seq2SeqCommitMessageGenerationModule"

    config["paths"] = {
        "root_dir": str(ROOT),
        "data_dir": str(ROOT / "data/") + "/",
        "log_dir": str(ROOT / "logs"),
        "output_dir": str(ROOT / "output"),
        "work_dir": str(ROOT),
    }

    # u can change batch size and number of workers here (not entirely necessary)
    config["data"]["batch_size"] = 16
    config["data"]["num_workers"] = 0

    # create module and data-module
    cfg = OmegaConf.create(config)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, _convert_="object")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _convert_="object")

    # loading checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))  # nosec B614
        model.load_state_dict(checkpoint["state_dict"])
    model = model.eval()

    # pass tokenizer from datamodule to model
    if hasattr(datamodule, "diff_tokenizer") and hasattr(datamodule, "msg_tokenizer"):
        model.diff_tokenizer = datamodule.diff_tokenizer
        model.msg_tokenizer = datamodule.msg_tokenizer
    elif hasattr(datamodule, "tokenizer"):
        model.tokenizer = datamodule.tokenizer
    else:
        logger.warning("No tokenizer param in datamodule. Maybe you need to specify a tokenizer")

    # let's take some batches from test dataloader for our demo
    datamodule.prepare_data()
    # noinspection PyArgumentList
    datamodule.setup()
    return model, datamodule


def generate_commit_message(model, diff_input: str, device: str = None):
    """Generate a commit message for a given diff input."""
    if device is None:
        device = next(model.parameters()).device

    # Prepare input based on available tokenizer
    if hasattr(model, "diff_tokenizer"):
        tokenized_input = model.diff_tokenizer(
            diff_input, return_tensors="pt", truncation=True, max_length=512
        )
    elif hasattr(model, "tokenizer"):
        tokenized_input = model.tokenizer(
            diff_input, return_tensors="pt", truncation=True, max_length=512
        )
    else:
        raise ValueError("No tokenizer found on model")

    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    # Create a Batch object
    batch = Batch(input_ids=input_ids, attention_mask=attention_mask)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(batch)
        # Use appropriate tokenizer for decoding
        if hasattr(model, "msg_tokenizer"):
            decoded_output = model.msg_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            decoded_output = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

def run_inference(model, datamodule, changes: list[dict[str, str]], device: str = None):
    diff_tokenizer = datamodule.diff_tokenizer
    line_sep  = datamodule.hparams.line_sep
    diff_max_len = datamodule.hparams.diff_max_len
    collator = DataCollatorWrapper(diff_tokenizer, False)

    diff =  _preprocess_mods(changes, line_sep)
    inputs = datamodule.diff_tokenizer(text=diff, max_length=diff_max_len, truncation=True)
    batch = collator([inputs])

    model = model.to(device=device)
    batch: Batch = batch.to(device)
    predictions = model.generate(batch)
    string_results = model._postprocess_generated(batch, predictions)
    df = pd.DataFrame(string_results)
    return df

def run_inference_llm(module, datamodule, changes: list[dict[str, str]]):
    batch = [
            map_example_to_request(
                {
                    "diff": _preprocess_mods(changes, datamodule.hparams.line_sep),
                },
                datamodule.completer,
                datamodule.hparams.system_content,
                datamodule.hparams.user_content_template,
                datamodule.hparams.max_prompt_token_count,
            )
        ]
    outputs = [item.content for item in module(batch)]
    df = pd.DataFrame([{**input, "prediction": output} for input, output in zip(batch, outputs)])
    return df

def run_inference_data_loader(model, loader, device: str = None):
    test_batches = []
    num_batches = 1
    for batch in loader:
        test_batches.append(batch)
        if num_batches is not None and len(test_batches) == num_batches:
            break

    # run inference
    with torch.no_grad():
        for batch in test_batches:
            batch: Batch = batch.to(device)
            predictions = model.generate(batch)
            string_results = model._postprocess_generated(batch, predictions)
            df = pd.DataFrame(string_results)
            print(df[["prediction", "target"]].head(n=16).to_string(max_colwidth=5000))

def fetch_git_changes(cwd: Path) -> list[dict[str, str]]:
    result = subprocess.run(["git", "diff", "--staged", "--name-status"], capture_output=True, text=True, cwd=cwd)

    def determine_change_type(change_type: str) -> str:
        change_types = {
            "A": "ADD",
            "M": "MODIFY",
            "D": "DELETE",
            "R": "RENAME",
            "C": "COPY",
        }
        if change_type in change_types:
            return change_types[change_type]
        if change_type.startswith("R"):
            return change_types["R"]
        raise Exception(f"Unknown change type {change_type}")

    modifications = []
    for line in result.stdout.splitlines():
        change, *filenames = line.split("\t")  # Split into type and filename
        filename = filenames[0]
        diff = subprocess.run(["git", "diff", "--staged", "--", filename], capture_output=True, text=True, cwd=cwd)
        item = {
            "change_type": determine_change_type(change),
            "old_path": filenames[0],
            "new_path": filenames[-1],
            "diff": _process_diff(diff.stdout, "\n")
        }
        modifications.append(item)
    return modifications

def _process_diff(diff: str, line_sep: str) -> str:
    """Processes a single diff (for a single file modification).

    Currently, it includes the following:
        * removing @@ ... @@ line – unnecessary git stuff
        * squeeze several whitespace sequences into one

    Args:
        diff: Input diff.
        line_sep: Newline separator that should be used in processed diff.

    Returns:
        Processed diff.
    """
    diff_lines = diff.split("\n")
    processed_lines = []

    for line in diff_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("@@") and line.endswith("@@"):
            processed_lines = []
        else:
            processed_lines.append(line)

    processed_diff = line_sep.join(processed_lines + [""])
    # squeeze several whitespace sequences into one (do not consider \n)
    processed_diff = re.sub(r"[^\S\n]+", " ", processed_diff)
    return processed_diff

def _preprocess_mods(mods: list[dict[str, str]], diff_line_sep: str) -> str:
    line_sep = diff_line_sep
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


if __name__ == "__main__":
    def main():
        model, datamodule = load_run(
            ROOT
            / "logs_2/train/runs/2025-01-24_23-12-54/checkpoints/epoch_023-val_MRR_top5_0.6524.ckpt"
        )
    main()
