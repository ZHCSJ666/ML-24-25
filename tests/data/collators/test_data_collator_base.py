import pytest
import torch
from transformers import AutoTokenizer

from src.data.components.collators.data_collator_base import DataCollatorBase
from src.data.types import SingleExample


@pytest.fixture(scope="module")
def default_tokenizers():
    """Fixture that initializes and returns default tokenizers for encoding and decoding.

    Both encoder and decoder tokenizers are initialized as CodeBERT tokenizers.
    """
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return encoder_tok, encoder_tok


@pytest.fixture(scope="module")
def collator_diff(default_tokenizers):
    """Returns an initialized collator utility instance."""
    encoder_tok, decoder_tok = default_tokenizers
    collator = DataCollatorBase(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        encoder_context_max_len=512,
        decoder_context_max_len=256,
        encoder_input_type="diff",
    )
    return collator


def test_diff_single_example(default_tokenizers, collator_diff):
    """Tests processing of a single "diff" input example to ensure correct padding, special tokens,
    and attention mask generation."""
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [
        SingleExample(
            diff_input_ids=[i for i in range(5, 105)],
            msg_input_ids=[],
            pos_in_file=0,
        )
    ]
    encoder_input_ids, encoder_attention_mask = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (1, 102)
    assert torch.all(
        encoder_input_ids
        == torch.tensor(
            [encoder_tok.bos_token_id] + [i for i in range(5, 105)] + [encoder_tok.eos_token_id]
        )
    )
    assert encoder_attention_mask.shape == (1, 102)
    assert torch.all(encoder_attention_mask == torch.tensor([1 for _ in range(100 + 2)]))


def test_diff_batch_pad_max_len(default_tokenizers, collator_diff):
    """Tests batch processing for "diff" inputs with padding to the maximum length, ensuring
    correct padding and attention mask handling in batch processing."""
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [
        SingleExample(
            diff_input_ids=[i for i in range(5, 105)],
            msg_input_ids=[],
            pos_in_file=0,
        ),
        SingleExample(
            diff_input_ids=[i for i in range(5, 50)],
            msg_input_ids=[],
            pos_in_file=1,
        ),
    ]
    encoder_input_ids, encoder_attention_mask = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (2, 102)
    assert torch.all(
        encoder_input_ids[0]
        == torch.tensor(
            [encoder_tok.bos_token_id] + [i for i in range(5, 105)] + [encoder_tok.eos_token_id]
        )
    )
    assert torch.all(
        encoder_input_ids[1]
        == torch.tensor(
            [encoder_tok.bos_token_id]
            + [i for i in range(5, 50)]
            + [encoder_tok.eos_token_id]
            + [encoder_tok.pad_token_id for _ in range(100 - 45)]
        )
    )
    assert encoder_attention_mask.shape == (2, 102)
    assert torch.all(encoder_attention_mask[0] == torch.tensor([1 for _ in range(100 + 2)]))
    assert torch.all(
        encoder_attention_mask[1]
        == torch.tensor([1 for _ in range(45 + 2)] + [0 for _ in range(100 - 45)])
    )


def test_diff_batch_truncate_max_len(default_tokenizers, collator_diff):
    """Tests batch processing of "diff" inputs that exceed the maximum allowed length, ensuring
    truncation is applied correctly to fit within the max encoder context length."""
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [
        SingleExample(
            diff_input_ids=[i for i in range(5, 1024)],
            msg_input_ids=[],
            pos_in_file=0,
        ),
        SingleExample(
            diff_input_ids=[i for i in range(5, 50)],
            msg_input_ids=[],
            pos_in_file=1,
        ),
    ]
    encoder_input_ids, encoder_attention_mask = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (2, 512)
    assert torch.all(
        encoder_input_ids[0]
        == torch.tensor(
            [encoder_tok.bos_token_id] + [i for i in range(5, 515)] + [encoder_tok.eos_token_id]
        )
    )
    assert torch.all(
        encoder_input_ids[1]
        == torch.tensor(
            [encoder_tok.bos_token_id]
            + [i for i in range(5, 50)]
            + [encoder_tok.eos_token_id]
            + [encoder_tok.pad_token_id for _ in range(510 - 45)]
        )
    )
    assert encoder_attention_mask.shape == (2, 512)
    assert torch.all(encoder_attention_mask[0] == torch.tensor([1 for _ in range(512)]))
    assert torch.all(
        encoder_attention_mask[1]
        == torch.tensor([1 for _ in range(45 + 2)] + [0 for _ in range(510 - 45)])
    )
