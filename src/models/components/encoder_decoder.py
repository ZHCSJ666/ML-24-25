import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.types import Batch


class EncoderDecoder(nn.Module):
    """A simple Transformer model for sequence-to-sequence tasks, consisting of an encoder and a
    decoder with token embeddings, positional encodings, and fully connected layers for output
    projection.

    Attributes:
        embed_size (int): The dimension of the embedding vectors.
        token_embedding (nn.Embedding): Embedding layer for tokenized input sequences.
        encoder_positional_encoding (nn.Parameter): Positional encoding matrix for token positions.
        decoder_positional_encoding (nn.Parameter): Positional encoding matrix for token positions.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder module.
        transformer_decoder (nn.TransformerDecoder): Transformer decoder module.
        fc_out (nn.Linear): Fully connected layer for projecting decoder output to vocabulary size.
        dropout (nn.Dropout): Dropout layer for regularization.

    Args:
        embed_size (int): Dimensionality of embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the feedforward layer.
        num_layers (int): Number of Transformer layers in the encoder and decoder.
        dropout (float): Dropout rate for regularization.
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(
        self,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
        encoder_vocab_size,
        decoder_vocab_size,
        encoder_context_max_len=5000,
        decoder_context_max_len=5000,
        num_beams=5,
        max_seq_length=512,
        beam_early_stopping=True,
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_seq_length = max_seq_length
        self.beam_early_stopping = beam_early_stopping
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.encoder_positional_encoding = nn.Parameter(
            self._generate_positional_encoding(encoder_context_max_len, embed_size),
            requires_grad=False,
        )
        self.decoder_positional_encoding = nn.Parameter(
            self._generate_positional_encoding(decoder_context_max_len, embed_size),
            requires_grad=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, decoder_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, embed_size):
        """Generate a positional encoding matrix using sine and cosine functions, providing a
        unique position-based encoding for each token up to max_len.

        Args:
            max_len (int): Maximum sequence length.
            embed_size (int): Dimensionality of each position's embedding vector.

        Returns:
            torch.Tensor: A tensor of shape (max_len, embed_size) containing positional encodings.
        """
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: Batch, src_mask=None, tgt_mask=None):
        """Defines the forward pass of the Transformer model, where the input sequence is passed
        through the encoder and decoder with positional encodings, and finally projected to output
        logits.

        Args:
            src (Batch): Input batch with encoder and decoder input IDs and attention masks.
            src_mask (torch.Tensor, optional): Mask for source padding tokens.
            tgt_mask (torch.Tensor, optional): Mask for target padding tokens.

        Returns:
            torch.Tensor: Output logits with shape (batch_size, target_sequence_length, vocab_size).
        """
        # Retrieve the source and target input IDs from the batch
        src_input_ids = src.encoder_input_ids
        tgt_input_ids = src.decoder_input_ids

        # Embed the tokens and add positional encoding
        src_embedded = self.token_embedding(src_input_ids)
        src_embedded += self.encoder_positional_encoding[: src_embedded.size(1), :].unsqueeze(0)
        src_embedded = self.dropout(src_embedded)

        tgt_embedded = self.token_embedding(tgt_input_ids)
        tgt_embedded += self.decoder_positional_encoding[: tgt_embedded.size(1), :].unsqueeze(0)
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
        log_probs = F.log_softmax(output, dim=2)
        return log_probs

    def generate(
        self,
        src: Batch,
    ) -> torch.Tensor:
        """Generates sequences using beam search decoding.

        Args:
            src (Batch): Input batch with encoder input IDs and attention masks.
            max_length (int, optional): Maximum length of the generated sequence. Defaults to 512.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            beam_early_stopping (bool, optional): Whether to stop early when all beams finish. Defaults to True.

        Returns:
            torch.Tensor: Tensor containing generated sequence token IDs with shape (batch_size, max_length).
        """

        self.eval()
        batch_size = src.encoder_input_ids.size(0)
        device = src.encoder_input_ids.device

        # Initialize beam search scores
        beam_scores = torch.zeros((batch_size, self.num_beams), device=device)
        beam_scores[:, 1:] = float("-inf")

        # Initialize decoder input with start token
        decoder_input = torch.zeros(
            (batch_size * self.num_beams, 1), dtype=torch.long, device=device
        )

        # Expand encoder output for beam search
        with torch.no_grad():
            # Forward pass through the encoder
            src_embed = self.token_embedding(
                src.encoder_input_ids
            ) + self.encoder_positional_encoding[: src.encoder_input_ids.size(1), :].unsqueeze(0)
            memory = self.transformer_encoder(
                src_embed,
                src_key_padding_mask=~src.encoder_attention_mask.bool(),
            )

            # Expand memory for beam search
            memory = (
                memory.unsqueeze(1)
                .expand(-1, self.num_beams, -1, -1)
                .reshape(batch_size * self.num_beams, -1, self.embed_size)
            )

            generated_ids = []
            done = [False] * batch_size

            # Generate tokens
            for step in range(self.max_seq_length):
                # Forward pass through decoder
                decoder_embed = self.token_embedding(
                    decoder_input
                ) + self.decoder_positional_encoding[: decoder_input.size(1), :].unsqueeze(0)
                decoder_output = self.transformer_decoder(
                    decoder_embed,
                    memory,
                    tgt_key_padding_mask=None,
                )

                # Get next token probabilities
                logits = self.fc_out(decoder_output[:, -1, :])
                next_token_logits = torch.log_softmax(logits, dim=-1)

                # Calculate scores for next tokens
                vocab_size = next_token_logits.shape[-1]
                next_scores = next_token_logits.view(batch_size, self.num_beams, -1)

                # Select top-k tokens with highest probability for each beam
                next_scores = next_scores + beam_scores.unsqueeze(-1)
                next_scores = next_scores.view(batch_size, -1)
                next_tokens = torch.topk(next_scores, self.num_beams, dim=1)

                # Update beam scores and tokens
                beam_scores = next_tokens.values
                beam_tokens = next_tokens.indices % vocab_size

                # Reshape beam tokens to match decoder input shape
                beam_tokens = beam_tokens.view(batch_size * self.num_beams, 1)

                # Concatenate along sequence dimension
                decoder_input = torch.cat([decoder_input, beam_tokens], dim=1)

                # Check for completed sequences
                if self.beam_early_stopping:
                    for batch_idx in range(batch_size):
                        if beam_tokens[batch_idx, 0] == src.decoder_input_ids[batch_idx, -1]:
                            done[batch_idx] = True
                    if all(done):
                        break

                generated_ids.append(beam_tokens[:, 0])  # Keep track of top beam only

            return torch.stack(generated_ids, dim=1)
