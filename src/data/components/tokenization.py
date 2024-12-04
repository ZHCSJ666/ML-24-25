from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(
    name_or_path: str,
    configuration: str = None,
) -> PreTrainedTokenizerFast:
    """Initializes tokenizer and adds special tokens when necessary."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer = add_special_tokens(tokenizer, configuration)
    return tokenizer


def add_special_tokens(
    tokenizer: PreTrainedTokenizerFast, preprocessor_configuration: Optional[str] = None
) -> PreTrainedTokenizerFast:
    """Adds special tokens to tokenizer based on preprocessor configuration.

    * pad_token is necessary for correct batch construction.
    * Several models employ additional special tokens in diffs representation.
    """
    if not tokenizer.pad_token:  # type: ignore[attr-defined]
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore[attr-defined]

    # TODO(ndersam): Remove usage of preprocessor_configuration
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
