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
        name_or_path: Name on HuggingFace hub or path to pretrained checkpoint.
        tokenizer: Tokenizer for the checkpoint (it's initialized earlier to add special tokens when necessary).
        diff_tokenizer: Tokenizer for source sequences (diffs)
        msg_tokenizer: Tokenizer for target sequences (messages)
        encoder_context_max_len: Maximum allowed input sequence length for encoder, used for initializing from scratch.
        decoder_context_max_len: Maximum allowed input sequence length for decoder, used for initializing from scratch.
        encoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize encoder.
        decoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize decoder.
        num_layers_encoder: If `encoder_name_or_path` is None, encoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, they will be uniformly picked.
        num_layers_decoder: If `decoder_name_or_path` is None, decoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, they will be uniformly picked.
        encoder_model_type: Optional – if encoder is initialized from scratch, this specific model class will be used.
        decoder_model_type: Optional – if decoder is initialized from scratch, this specific model class will be used.
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
            model.init_weights()
            self.model = model
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

    def generate(self, batch: Batch, max_length=64, **generation_kwargs) -> Any:
        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            max_length=max_length,
            # decoder_input_ids=batch.decoder_input_ids,
            # decoder_attention_mask=batch.decoder_attention_mask,
            **generation_kwargs,
        )

    @staticmethod
    def get_decoder_start_token_id(name_or_path):
        config = AutoConfig.from_pretrained(name_or_path)
        return config.decoder_start_token_id
