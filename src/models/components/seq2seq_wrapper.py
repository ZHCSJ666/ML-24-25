from typing import Any, Optional

import torch.nn as nn
from transformers import (
    AutoConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    PreTrainedModel,
)

from src.data.types import Batch


class Seq2SeqWrapper(nn.Module):
    """This class serves as a wrapper of Transformer-based models for commit message completion
    task.

    More specifically, this class relies on pretrained seq2seq models from HuggingFace Transformers.

    Args:
        model: Tokenizer for target sequences (messages)
        encoder_context_max_len: Maximum allowed input sequence length for encoder, used for initializing from scratch.
        decoder_context_max_len: Maximum allowed input sequence length for decoder, used for initializing from scratch.
        encoder: Optional – name or path to pretrained checkpoint to initialize encoder.
        decoder: Optional – name or path to pretrained checkpoint to initialize decoder.
        tie_encoder_decoder: If set to `True`, encoder and decoder will share the same parameters
          (should be used with the same model classes and tokenizers).
        tie_word_embeddings: If set to `True`, encoder and decoder will share the same parameters for embedding layers
          (should be used with the same model classes and tokenizers).
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        encoder_context_max_len=None,
        decoder_context_max_len=None,
        encoder_vocab_size: Optional[int] = None,
        decoder_vocab_size: Optional[int] = None,
        tie_encoder_decoder: Optional[bool] = None,
        tie_word_embeddings: Optional[bool] = None,
    ):
        super().__init__()
        if model is not None:
            if decoder_vocab_size == encoder_vocab_size:
                model.resize_token_embeddings(encoder_vocab_size)
            else:
                model.encoder.resize_token_embeddings(encoder_vocab_size)
                model.decoder.resize_token_embeddings(decoder_vocab_size)
                model.lm_head = nn.Linear(
                    in_features=model.lm_head.in_features, out_features=decoder_vocab_size
                )
            self.model = model
            model.apply(model._init_weights)
        else:
            encoder.resize_token_embeddings(decoder_vocab_size)
            decoder.resize_token_embeddings(decoder_vocab_size)
            config = EncoderDecoderConfig.from_encoder_decoder_configs(
                encoder_config=encoder.config, decoder_config=decoder.config  # type: ignore[attr-defined]
            )
            if tie_encoder_decoder is not None:
                config.encoder.tie_encoder_decoder = tie_encoder_decoder
                config.decoder.tie_encoder_decoder = tie_encoder_decoder
                config.tie_encoder_decoder = tie_encoder_decoder
            if tie_word_embeddings is not None:
                config.tie_word_embeddings = tie_word_embeddings

            self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    def forward(self, batch: Batch) -> Any:
        return self.model(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        )

    def generate(
        self,
        batch: Batch,
        max_length=200,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=2.5,
        length_penalty=1.0,
        **generation_kwargs,
    ) -> Any:
        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            # decoder_input_ids=batch.decoder_input_ids,
            # decoder_attention_mask=batch.decoder_attention_mask,
            **generation_kwargs,
        )

    @staticmethod
    def get_decoder_start_token_id(name_or_path):
        config = AutoConfig.from_pretrained(name_or_path)
        return config.decoder_start_token_id
