# models/train.py

"""
Model Training Module
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from models.model import get_model
from utils.utils import CommitDataset
from tqdm import tqdm

def train_model(config):
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['model_name'])
    model = get_model(config['model']['model_name'])

    # Prepare dataset
    train_dataset = CommitDataset(config['data']['train_file'], tokenizer, config['training']['max_seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Start training
    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config['training']['epochs']}"):
            inputs = batch['input_ids']
            labels = batch['labels']
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader)}")

    # Save the model
    model.save_pretrained(config['model']['save_dir'])
    print("Model training completed and saved.")
