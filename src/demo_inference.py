import os
from pathlib import Path

import hydra
import pandas as pd
import rootutils
import torch
from huggingface_hub import HfApi, create_repo, login
from lightning import LightningDataModule, LightningModule
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)

from src.data.types import Batch


def load_run(checkpoint_path: Path):
    config_path = checkpoint_path.parent.parent / ".hydra/config.yaml"
    config = OmegaConf.load(config_path)

    # I renamed some classes after training this checkpoint, hence I'm updating the old names here
    config.data._target_ = "src.data.CommitChronicleSeq2SeqDataModule"
    config.model._target_ = "src.models.Seq2SeqCommitMessageGenerationModule"

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":

        checkpoint = torch.load(checkpoint_path)  # nosec B614
    else:

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))  # nosec B614

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    # pass tokenizer from datamodule to model
    if hasattr(datamodule, "diff_tokenizer") and hasattr(datamodule, "msg_tokenizer"):
        model.diff_tokenizer = datamodule.diff_tokenizer
        model.msg_tokenizer = datamodule.msg_tokenizer
    elif hasattr(datamodule, "tokenizer"):
        model.tokenizer = datamodule.tokenizer
    else:
        raise ValueError("tokenizer not set")

    # let's take some batches from test dataloader for our demo
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.test_dataloader()
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


if __name__ == "__main__":
    model, datamodule = load_run(
        ROOT
        / "logs_2/train/runs/2025-01-24_23-12-54/checkpoints/epoch_023-val_MRR_top5_0.6524.ckpt"
    )
