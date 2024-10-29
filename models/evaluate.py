# models/evaluate.py

"""
Model Evaluation Module
"""

from transformers import GPT2Tokenizer
from models.model import get_model
from utils.utils import CommitDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate_model(config):
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['model_name'])
    model = get_model(config['model']['model_name'])
    model.load_state_dict(torch.load(config['model']['save_dir'] + '/pytorch_model.bin'))
    model.eval()

    # Prepare dataset
    test_dataset = CommitDataset(config['data']['test_file'], tokenizer, config['training']['max_seq_length'])
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Define evaluation metrics
    # ...

    # Start evaluation
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['input_ids']
            # Generate commit message
            outputs = model.generate(input_ids=inputs)
            # Calculate evaluation metrics
            # ...

    print("Model evaluation completed.")
