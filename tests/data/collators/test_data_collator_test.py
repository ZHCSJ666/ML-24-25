import pytest
import torch
from transformers import AutoTokenizer

from src.data.components.collators import DataCollatorTest
from src.data.types import SingleExample


@pytest.fixture
def default_tokenizers():
    """Return default tokenizers for testing.

    Returns:
        tuple: A tuple containing (encoder_tokenizer, decoder_tokenizer)
    """
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")  # nosec B106
    return encoder_tok, encoder_tok


@pytest.mark.parametrize(
    "msgs",
    [
        (["Residence certainly elsewhere something she preferred cordially law."],),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
        ),
    ],
)
def test_decoder_input(msgs, default_tokenizers):
    """Test the decoder input processing functionality.

    Args:
        msgs: Test messages to process
        default_tokenizers: Fixture providing tokenizers
    """
    encoder_tok, decoder_tok = default_tokenizers
    inputs = []
    for i, msg in enumerate(msgs):
        msgs_ids = decoder_tok(
            msg, add_special_tokens=False, padding=False, truncation=False
        ).input_ids
        inputs.append(
            SingleExample(
                diff_input_ids=[],
                msg_input_ids=msgs_ids,
                pos_in_file=i,
            )
        )

    for context_ratio in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        data_collator = DataCollatorTest(
            diff_bos_token_id=encoder_tok.bos_token_id,
            diff_eos_token_id=encoder_tok.eos_token_id,
            diff_pad_token_id=encoder_tok.pad_token_id,
            msg_bos_token_id=decoder_tok.bos_token_id,
            msg_eos_token_id=decoder_tok.eos_token_id,
            msg_pad_token_id=decoder_tok.pad_token_id,
            msg_sep_token_id=decoder_tok.sep_token_id,
            diff_tokenizer=encoder_tok,
            msg_tokenizer=decoder_tok,
            decoder_context_max_len=256,
            context_ratio=context_ratio,
            encoder_input_type="diff",
            encoder_context_max_len=None,
        )
        (
            decoder_input_ids,
            decoder_attention_mask,
            targets,
            prefixes,
        ) = data_collator._process_decoder_input(inputs)

        # check left-side padding
        for message, mask in zip(decoder_input_ids, decoder_attention_mask):
            if (mask == 0).nonzero().numel():
                assert mask[0] == 0
                assert torch.all(mask[: (mask == 0).nonzero().squeeze()[-1] + 1] == 0)
                assert torch.all(mask[(mask == 0).nonzero().squeeze()[-1] + 1 :] == 1)
                assert torch.all(
                    message[: (mask == 0).nonzero().squeeze()[-1] + 1] == decoder_tok.pad_token_id
                )
                assert torch.all(
                    message[(mask == 0).nonzero().squeeze()[-1] + 1] == decoder_tok.bos_token_id
                )

        for context, prefix, target, msg in zip(decoder_input_ids, prefixes, targets, msgs):
            decoded_context = decoder_tok.decode(context, skip_special_tokens=True)
            assert decoded_context + prefix + target == msg


def test_process_msg_gen(default_tokenizers):
    """Test message generation processing.

    Args:
        default_tokenizers: Fixture providing tokenizers
    """
    encoder_tok, decoder_tok = default_tokenizers

    data_collator = DataCollatorTest(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        diff_tokenizer=encoder_tok,
        msg_tokenizer=decoder_tok,
        encoder_context_max_len=None,
        decoder_context_max_len=None,
        context_ratio=None,
        encoder_input_type=None,
    )

    message = "Simple message example"
    message_ids = decoder_tok(
        message, add_special_tokens=False, padding=False, truncation=False
    ).input_ids

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=2
    )
    assert msg_input_ids == []
    assert target == "mple message example"
    assert prefix == "Si"

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=6
    )
    assert msg_input_ids == []
    assert target == " message example"
    assert prefix == "Simple"

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=7
    )
    assert (
        msg_input_ids
        == decoder_tok(
            "Simple ", add_special_tokens=False, padding=False, truncation=False
        ).input_ids
    )
    assert target == "message example"
    assert prefix == ""

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=9
    )
    assert (
        msg_input_ids
        == decoder_tok(
            "Simple", add_special_tokens=False, padding=False, truncation=False
        ).input_ids
    )
    assert target == "ssage example"
    assert prefix == " me"

    message = "chore(deps): update version to v1.0.0-SNAPSHOT"
    message_ids = decoder_tok(
        message, add_special_tokens=False, padding=False, truncation=False
    ).input_ids

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=8
    )
    assert msg_input_ids == []
    assert target == "ps): update version to v1.0.0-SNAPSHOT"
    assert prefix == "chore(de"

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=12
    )
    assert msg_input_ids == []
    assert target == " update version to v1.0.0-SNAPSHOT"
    assert prefix == "chore(deps):"

    msg_input_ids, target, prefix = data_collator._process_msg_gen(
        message_ids=message_ids, context_len=34
    )
    assert (
        msg_input_ids
        == decoder_tok(
            "chore(deps): update version to",
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
    )
    assert target == "0.0-SNAPSHOT"
    assert prefix == " v1."
