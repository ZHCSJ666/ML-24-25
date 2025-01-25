import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerFast


def update_tokenization_properties(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    decoder_tokenizer: PreTrainedTokenizerFast | None = None,
) -> PreTrainedModel:
    if decoder_tokenizer is None:
        decoder_tokenizer = tokenizer

    # update token embeddings
    encoder_vocab_size = len(tokenizer)
    decoder_vocab_size = len(decoder_tokenizer)
    if decoder_vocab_size == encoder_vocab_size:
        model.resize_token_embeddings(encoder_vocab_size)
    else:
        model.encoder.resize_token_embeddings(encoder_vocab_size)
        model.decoder.resize_token_embeddings(decoder_vocab_size)
        model.lm_head = nn.Linear(
            in_features=model.lm_head.in_features, out_features=decoder_vocab_size
        )

    decoder_start_token_id = model.generation_config.decoder_start_token_id
    is_decoder_start_token_pad_token = decoder_start_token_id == model.config.pad_token_id
    is_decoder_start_token_bos_token = decoder_start_token_id == model.config.bos_token_id
    is_decoder_start_token_eos_token = decoder_start_token_id == model.config.eos_token_id

    # update config object
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size

    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # update decoder_start_token_id
    if is_decoder_start_token_pad_token:
        model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
    elif is_decoder_start_token_bos_token:
        model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    elif is_decoder_start_token_eos_token:
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    else:
        raise ValueError("Unknown decoder start token type")

    return model
