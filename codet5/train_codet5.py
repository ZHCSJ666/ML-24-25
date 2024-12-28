#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_codet5.py: Fine-tune Salesforce/codet5-small on the Rissou/simplified-commit-chronicle dataset
to generate commit messages, with a custom callback for visualization.
Now includes type-check for 'mods' and 'message' to handle ValueError from tokenizer.
"""

import argparse
import random
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CodeT5 on simplified-commit-chronicle with visualization")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-small",
                        help="Which CodeT5 model to use (e.g. codet5-small/codet5-base).")
    parser.add_argument("--output_dir", type=str, default="./codet5-commit-generation",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--max_source_length", type=int, default=256,
                        help="Max token length for input (mods).")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="Max token length for output (message).")
    args = parser.parse_args()
    return args


class LossVisualizationCallback(TrainerCallback):
    """
    Custom callback to record and visualize training and evaluation losses.
    """

    def __init__(self, logging_steps=100):
        super().__init__()
        self.logging_steps = logging_steps
        self.train_loss_history = []
        self.eval_loss_history = []
        self.global_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            train_loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            step = state.global_step

            if train_loss is not None:
                self.train_loss_history.append(train_loss)
                self.global_steps.append(step)
            if eval_loss is not None:
                self.eval_loss_history.append((step, eval_loss))

    def on_train_end(self, args, state, control, **kwargs):
        # Plot the recorded losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_steps, self.train_loss_history, label='Training Loss', color='blue')

        if self.eval_loss_history:
            eval_steps, eval_losses = zip(*self.eval_loss_history)
            plt.plot(eval_steps, eval_losses, label='Validation Loss', color='red', marker='o')

        plt.title("Training & Validation Loss over Steps")
        plt.xlabel("Global Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(args.output_dir + "/loss_curve.png")
        plt.show()
        print(f"Loss curve saved to {args.output_dir}/loss_curve.png")


def main():
    args = parse_args()

    # 1. Load dataset
    dataset = load_dataset("Rissou/simplified-commit-chronicle")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    # test_dataset = dataset["test"]

    # 2. Load tokenizer & model with auto classes
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 3. Define the tokenize function
    def tokenize_fn(examples):
        # Raw data from 'mods' and 'message'
        raw_mods = examples["mods"]
        raw_message = examples["message"]

        # ---- Convert 'mods' to a string or list[str] ----
        # If raw_mods is already a list of strings, that's fine.
        # Otherwise, convert to string. 
        # Because we're using "batched=True", raw_mods might be a list of items (batch dimension).
        # We'll do a list comprehension to ensure each item is a string.
        if isinstance(raw_mods, list):
            # Here raw_mods is a list corresponding to the batch. Each item could be str/dict/list.
            # We'll stringify each item individually.
            mods_list = []
            for item in raw_mods:
                if isinstance(item, str):
                    mods_list.append(item)
                else:
                    # Convert non-string item to string
                    mods_list.append(str(item))
        else:
            # If for some reason raw_mods is a single item, convert to list with one element
            mods_list = [str(raw_mods)]

        # ---- Convert 'message' to a string or list[str] ----
        if isinstance(raw_message, list):
            message_list = []
            for msg in raw_message:
                if isinstance(msg, str):
                    message_list.append(msg)
                else:
                    message_list.append(str(msg))
        else:
            message_list = [str(raw_message)]

        # Now do tokenization
        model_inputs = tokenizer(
            mods_list,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True
        )
        labels = tokenizer(
            message_list,
            max_length=args.max_target_length,
            padding="max_length",
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 4. Map / preprocess
    train_dataset_tokenized = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=train_dataset.column_names
    )
    val_dataset_tokenized = val_dataset.map(
        tokenize_fn, batched=True, remove_columns=val_dataset.column_names
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to="none",
        fp16=False,
        push_to_hub=False
    )

    # 6. Trainer with custom visualization callback
    visualization_callback = LossVisualizationCallback(logging_steps=training_args.logging_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        callbacks=[visualization_callback]
    )

    # 7. Train
    trainer.train()

    # 8. Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
