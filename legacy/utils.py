# utils/utils.py

"""
Utility Functions Module
"""

from torch.utils.data import Dataset
import json

class CommitDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        code_diff = sample['code_diff']
        commit_message = sample['commit_message']
        # Encode inputs and labels
        inputs = self.tokenizer.encode_plus(
            code_diff,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        labels = self.tokenizer.encode_plus(
            commit_message,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }
