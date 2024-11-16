import logging
from copy import deepcopy
from typing import Optional, Tuple

import hydra
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizers(
    msg_tokenizer_name_or_path: str,
    diff_tokenizer_name_or_path: Optional[str],
    configuration: str = None,
) -> Tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    """Initializes tokenizers and adds special tokens when necessary."""
    try:
        msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)
    except ValueError:
        msg_tokenizer = AutoTokenizer.from_pretrained(
            hydra.utils.to_absolute_path(msg_tokenizer_name_or_path)
        )

    msg_tokenizer = add_special_tokens(msg_tokenizer, configuration)

    if not diff_tokenizer_name_or_path:
        logging.warning("Diff tokenizer is not set, using message tokenizer as a default")
        diff_tokenizer = deepcopy(msg_tokenizer)
    elif diff_tokenizer_name_or_path == msg_tokenizer_name_or_path:
        diff_tokenizer = deepcopy(msg_tokenizer)
    else:
        try:
            diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)
        except ValueError:
            diff_tokenizer = AutoTokenizer.from_pretrained(
                hydra.utils.to_absolute_path(diff_tokenizer_name_or_path)
            )
        diff_tokenizer = add_special_tokens(diff_tokenizer, configuration)

    return diff_tokenizer, msg_tokenizer


def add_special_tokens(
    tokenizer: PreTrainedTokenizerFast, preprocessor_configuration: str
) -> PreTrainedTokenizerFast:
    """Adds special tokens to tokenizer based on preprocessor configuration.

    * sep_token is necessary for correct history construction.
    * pad_token is necessary for correct batch construction.
    * Several models employ additional special tokens in diffs representation.
    """
    if not tokenizer.sep_token:  # type: ignore[attr-defined]
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})  # type: ignore[attr-defined]
    if not tokenizer.pad_token:  # type: ignore[attr-defined]
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore[attr-defined]

    if preprocessor_configuration == "codereviewer":
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<add>", "<del>", "<keep>"]}
        )  # type: ignore[attr-defined]
    elif preprocessor_configuration == "race":
        tokenizer.add_special_tokens(  # type: ignore[attr-defined]
            {
                "additional_special_tokens": [
                    "<KEEP>",
                    "<KEEP_END>",
                    "<INSERT>",
                    "<INSERT_END>",
                    "<DELETE>",
                    "<DELETE_END>",
                    "<REPLACE_OLD>",
                    "<REPLACE_NEW>",
                    "<REPLACE_END>",
                ]
            }
        )
    return tokenizer
