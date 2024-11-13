import torch
import torch.nn as nn
import numpy as np
from src.data.types import Batch


class SimpleTransformer(nn.Module):
    def __init__(
        self, embed_size, num_heads, hidden_dim, num_layers, dropout, vocab_size
    ):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(5000, embed_size), requires_grad=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size)
        )
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: Batch, src_mask=None, tgt_mask=None):
        # Retrieve the source and target input IDs from the batch
        src_input_ids = src.encoder_input_ids
        tgt_input_ids = src.decoder_input_ids

        # Embed the tokens and add positional encoding
        src_embedded = self.token_embedding(src_input_ids)
        src_embedded += self.positional_encoding[: src_embedded.size(1), :].unsqueeze(0)
        src_embedded = self.dropout(src_embedded)

        tgt_embedded = self.token_embedding(tgt_input_ids)
        tgt_embedded += self.positional_encoding[: tgt_embedded.size(1), :].unsqueeze(0)
        tgt_embedded = self.dropout(tgt_embedded)

        # Apply the Transformer encoder and decoder
        memory = self.transformer_encoder(
            src_embedded, src_key_padding_mask=~src.encoder_attention_mask.bool()
        )
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_key_padding_mask=~src.decoder_attention_mask.bool(),
        )

        # Project the decoder outputs to vocabulary size
        output = self.fc_out(output)

        return output

    def generate(
        self,
        src: Batch,
        max_length: int = 512,
        num_beams: int = 5,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        self.eval()
        batch_size = src.encoder_input_ids.size(0)
        device = src.encoder_input_ids.device

        # Initialize the decoder input with the start token
        start_token = torch.tensor(
            [[self.token_embedding.padding_idx]], device=device
        ).repeat(batch_size, 1)
        decoder_input = start_token

        generated_ids = []
        with torch.no_grad():
            # Forward pass through the encoder
            memory = self.transformer_encoder(
                self.token_embedding(src.encoder_input_ids)
                + self.positional_encoding[
                    : src.encoder_input_ids.size(1), :
                ].unsqueeze(0),
                src_key_padding_mask=~src.encoder_attention_mask.bool(),
            )

            # Iteratively generate the output sequence
            for _ in range(max_length):
                # Forward pass through the decoder
                output = self.transformer_decoder(
                    self.token_embedding(decoder_input)
                    + self.positional_encoding[: decoder_input.size(1), :].unsqueeze(0),
                    memory,
                    tgt_key_padding_mask=~(
                        decoder_input != self.token_embedding.padding_idx
                    ).bool(),
                )
                output = self.fc_out(output[:, -1, :])

                # Select the next token
                next_token = output.argmax(dim=-1)
                generated_ids.append(next_token)
                decoder_input = torch.cat(
                    [decoder_input, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(generated_ids, dim=1)
