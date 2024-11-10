from src.data.types import Batch
import torch
import torch.nn as nn
import numpy as np

class SimpleTransformer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, num_layers, dropout, vocab_size):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(5000, embed_size), requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: Batch, src_mask=None):
        src = self.token_embedding(src.retrieved_diff_input_ids) + self.positional_encoding[:src.retrieved_diff_input_ids.size(1), :]
        src = self.dropout(src)
        memory = self.transformer_encoder(src.permute(1, 0, 2), src_mask)
        output = self.fc_out(memory.permute(1, 0, 2))
        return output