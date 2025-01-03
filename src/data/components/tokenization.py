from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(
    name_or_path: str, task: str | None = None, **kwargs
) -> PreTrainedTokenizerFast:
    """Initializes tokenizer and adds special tokens when necessary."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    tokenizer = add_special_tokens(tokenizer, task)
    return tokenizer


def add_special_tokens(tokenizer: PreTrainedTokenizerFast, task) -> PreTrainedTokenizerFast:
    """Adds special tokens to tokenizer based on preprocessor configuration.

    * pad_token is necessary for correct batch construction.
    * Several models employ additional special tokens in diffs representation.
    """

    # custom logic for causal language modeling (clm)
    if task == "clm":
        # Most LLMs don't have a pad token by default
        # See https://huggingface.co/docs/transformers/en/llm_tutorial#common-pitfalls
        tokenizer.pad_token = tokenizer.eos_token

        # this token is only used for decoder-only models
        # it's used to construct input in the form `<git diff> <|sep|> <commit message>`
        if not tokenizer.sep_token or tokenizer.sep_token == tokenizer.eos_token:  # type: ignore[attr-defined]
            tokenizer.add_special_tokens({"sep_token": "<|sep|>"})  # type: ignore[attr-defined]

        return tokenizer

    if not tokenizer.pad_token:  # type: ignore[attr-defined]
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})  # type: ignore[attr-defined]

    return tokenizer
