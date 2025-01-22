#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

def main():

    dataset = load_dataset("Rissou/simplified-commit-chronicle")
    test_dataset = dataset["test"]

 
    model_path = "./codet5-commit-generation" 
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    random_idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[random_idx]

    mods_text = sample["mods"]
    golden_message = sample["message"]

    print("=== Mods (Input) ===")
    print(mods_text)
    print("\n=== Golden Commit Message ===")
    print(golden_message)


    max_source_length = 256
    max_target_length = 128

    input_ids = tokenizer(
        mods_text,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True
    )

    pred_message = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Predicted Commit Message ===")
    print(pred_message)

if __name__ == "__main__":
    main()
