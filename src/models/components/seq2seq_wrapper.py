from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput
from loguru import logger
from src.data.types import Batch
from src.models.components.utils import update_tokenization_properties


class Seq2SeqWrapper(nn.Module):
    """This class serves as a wrapper of Transformer-based models for commit message completion
    task.

    More specifically, this class relies on pretrained seq2seq models from HuggingFace Transformers.

    Args:
        model: Tokenizer for target sequences (messages)
        encoder_context_max_len: Maximum allowed input sequence length for encoder, used for initializing from scratch.
        decoder_context_max_len: Maximum allowed input sequence length for decoder, used for initializing from scratch.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        encoder_context_max_len=None,
        decoder_context_max_len=None,
        encoder_vocab_size: Optional[int] = None,
        decoder_vocab_size: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        decoder_tokenizer: Optional[PreTrainedTokenizerFast] = None,
        init_weights: bool = False,
    ):
        super().__init__()
        if tokenizer is not None:
            model = update_tokenization_properties(model, tokenizer, decoder_tokenizer)
        elif encoder_vocab_size is not None and decoder_vocab_size is not None:
            logger.warning(
                "Passing `encoder_vocab_size` and `decoder_vocab_size` will be deprecated. Use `tokenizer` and `decoder_tokenizer` params instead."
            )
            if decoder_vocab_size == encoder_vocab_size:
                model.resize_token_embeddings(encoder_vocab_size)
            else:
                model.encoder.resize_token_embeddings(encoder_vocab_size)
                model.decoder.resize_token_embeddings(decoder_vocab_size)
                model.lm_head = nn.Linear(
                    in_features=model.lm_head.in_features, out_features=decoder_vocab_size
                )
        if init_weights:
            # noinspection PyProtectedMember
            model.apply(model._init_weights)
        self.model = model

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        output: Seq2SeqLMOutput = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        )
        return {"logits": output.logits, "loss": output.loss}

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
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            # decoder_input_ids=batch.decoder_input_ids,
            # decoder_attention_mask=batch.decoder_attention_mask,
            **generation_kwargs,
        )
