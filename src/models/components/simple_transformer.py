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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
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
        # Retrieve the source input IDs from the batch
        src_input_ids = src.encoder_input_ids  # (batch_size, seq_len)
        
        # Embed the tokens and add positional encoding
        src_embedded = self.token_embedding(src_input_ids)  # (batch_size, seq_len, embed_size)
        src_embedded += self.positional_encoding[:src_embedded.size(1), :].unsqueeze(0)  # Broadcasting
        src_embedded = self.dropout(src_embedded)
        
        # Apply the Transformer encoder
        memory = self.transformer_encoder(src_embedded, src_key_padding_mask=~src.encoder_attention_mask.bool())  # (batch_size, seq_len, embed_size)
        
        # Project the encoder outputs to vocabulary size
        output = self.fc_out(memory)  # (batch_size, seq_len, vocab_size)
        return output

    def generate(self, src: Batch, max_length: int = 512, num_beams: int = 5, early_stopping: bool = True) -> torch.Tensor:
        """Generates sequences for the given source inputs using greedy decoding."""
        self.eval()
        generated_ids = []
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(src)  # (batch_size, seq_len, vocab_size)
            # For simplicity, use greedy decoding by selecting the argmax
            generated_ids = torch.argmax(outputs, dim=-1)  # (batch_size, seq_len)
        return generated_ids
