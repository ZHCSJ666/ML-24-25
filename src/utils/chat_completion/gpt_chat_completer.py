import json
import math
import multiprocessing as mp
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Any

import tiktoken
from datasets import Dataset, concatenate_datasets
from loguru import logger
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from transformers import AutoTokenizer

from .chat_completer import LLMChatCompleter, LLMChatCompleterResponse
from ..more_utils import load_jsonl_as_dataset


class GPTChatCompleter(LLMChatCompleter):
    """
    A chat completion handler for OpenAI's GPT models.

    This class facilitates interactions with OpenAI's GPT API for chat-based
    completions. It supports token counting, batch processing, and cost estimation.

    Attributes:
        PRICING_PER_1M_TOKENS (dict): A dictionary mapping supported models to their
            pricing per million tokens.
        api_key (str): The OpenAI API key for authentication.
        model (str): The name of the GPT model to use.
        temperature (float): The temperature setting for response generation.
        max_prompt_token_count (int): Maximum token count allowed for the prompt.
        max_response_token_count (int): Maximum token count allowed for the response.
        num_proc (int): Number of parallel processes for batch processing.
        batch_limit_tpd (int): Token per day limit for batch processing.
        batch_size (int): The number of examples per batch, derived from batch_limit_tpd.
    """

    PRICING_PER_1M_TOKENS = {
        "gpt-4o-mini-2024-07-18": 0.075,
        "deepseek-chat": 0.14,
    }

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_prompt_token_count: int,
        max_response_token_count: int,
        batch_limit_tpd: int,
        tokenizer: str | None = None,
        base_url: str | None = None,
        num_proc: int | None = None,
    ) -> None:
        """
        Initializes the GPTChatCompleter.

        Args:
            api_key (str): The OpenAI API key for authentication.
            model (str): The GPT model name (e.g., "gpt-4o-mini-2024-07-18").
            temperature (float): Controls randomness in the model’s responses (higher values = more randomness).
            max_prompt_token_count (int): The maximum number of tokens allowed in the input prompt.
            max_response_token_count (int): The maximum number of tokens allowed in the model's response.
            batch_limit_tpd (int): The maximum number of tokens that can be processed per day in batch mode.
            num_proc (int | None, optional): Number of processes to use for batch operations. Defaults to available CPU count.

        Raises:
            AssertionError: If `api_key` is None or `model` is not supported.
        """
        assert api_key is not None
        assert (
            model in GPTChatCompleter.PRICING_PER_1M_TOKENS
        ), f"Model '{model}' is not supported yet. Available models include [{list(GPTChatCompleter.PRICING_PER_1M_TOKENS.keys())}]'"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else None
        self.temperature = temperature
        self.max_prompt_token_count = max_prompt_token_count
        self.max_response_token_count = max_response_token_count
        self.num_proc = num_proc or mp.cpu_count()

        # batch api stuff
        self.batch_limit_tpd = batch_limit_tpd
        self.batch_size = batch_limit_tpd // max_prompt_token_count

    # written here to allow serialization during multiprocessing
    @property
    def client(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def count_tokens(self, message: str | list[dict[str, str]]) -> int:
        if self.tokenizer is not None:
            if not isinstance(message, str):
                try:
                    message = self.tokenizer.apply_chat_template(
                        message,
                        truncation=False,
                        tokenize=False,
                        add_generation_prompt=True,
                        max_length=sys.maxsize,  # to suppress a warning
                    )
                except ValueError:
                    messages = []
                    for m in message:
                        for k, v in m.items():
                            messages.append(v)
                    message = " ".join(messages)
            return len(self.encode(message))
        if isinstance(message, str):
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(message))
        return num_tokens_from_messages(message, model=self.model)

    def encode(self, message: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer(
                [message],
                truncation=False,
                # to suppress a warning
                max_length=sys.maxsize,
            )["input_ids"][0]
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.encode(message)

    def decode(self, message: list[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(message, skip_special_tokens=True)
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.decode(message)

    # https://cookbook.openai.com/examples/how_to_handle_rate_limits
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def complete_chat(self, messages: list[dict[str, str]]) -> LLMChatCompleterResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_response_token_count,
        )
        return LLMChatCompleterResponse(
            content=response.choices[0].message.content,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )

    def prepare_batch(self, dataset: Dataset, working_dir: Path) -> None:
        assert "system_content" in dataset.column_names
        assert "user_content" in dataset.column_names

        requests_dir = working_dir / "requests"
        batch_files = [
            requests_dir / f"{i:06d}.jsonl" for i in range(0, len(dataset), self.batch_size)
        ]
        if all(file.exists() for file in batch_files):
            logger.info("Batch already processed")
            return

        def map_function(example, model: str, max_response_token_count: int, temperature: float):
            return {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": example["system_content"],
                        },
                        {
                            "role": "user",
                            "content": example["user_content"],
                        },
                    ],
                    "max_tokens": max_response_token_count,
                    "temperature": temperature,
                },
            }

        dataset = dataset.map(
            map_function,
            fn_kwargs={
                "model": self.model,
                "max_response_token_count": self.max_response_token_count,
                "temperature": self.temperature,
            },
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )

        requests_dir.mkdir(exist_ok=True, parents=True)

        # split big batch into mini-batches so as not to create a large batch file that will make ChatGPT unhappy
        for i in tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[i : i + self.batch_size]
            batch = Dataset.from_dict(batch)
            batch.to_json(requests_dir / f"{i:06d}.jsonl", orient="records", lines=True)

    def submit_batch(self, working_dir: Path) -> Path:
        requests_dir = working_dir / "requests"
        responses_dir = working_dir / "responses"
        output_dataset_path = working_dir / "results.jsonl"

        if output_dataset_path.exists():
            return output_dataset_path
        batch_request_files = [
            path for path in requests_dir.iterdir() if path.is_file() and path.suffix == ".jsonl"
        ]
        batch_request_files = sorted(batch_request_files)
        batch_response_files = [responses_dir / path.name for path in batch_request_files]

        for request_file, response_file in tqdm(
            zip(batch_request_files, batch_response_files)
        ):  # type: Path, Path

            # create batch job and save batch_id
            batch_id_file = request_file.with_suffix(".txt")
            if not batch_id_file.exists():
                batch_id = submit_chat_completion_job(self.client, request_file)
                with open(batch_id_file, "w") as f:
                    f.write(batch_id)
            else:
                with open(batch_id_file, "r") as f:
                    batch_id = f.read()

            # wait for batch job completion
            is_completed, api_status, api_errors = wait_for_chat_completion_job(
                self.client, batch_id, response_file
            )

            # we want to complete current file before moving to the next
            # to not exceed OpenAI's API batch limit request
            if not is_completed:
                logger.error(f"Could not complete chat completion job for '{request_file}'.")
                logger.error(f"{api_status}, {api_errors}")
                raise Exception("Could not complete chat completion job.")

        # create new dataset from responses
        datasets = []
        for response_file in batch_response_files:
            dataset = load_jsonl_as_dataset(response_file)
            dataset = dataset.map(
                lambda example: {
                    "message": example["response"]["body"]["choices"][0]["message"]["content"]
                },
                num_proc=self.num_proc,
                remove_columns=dataset.column_names,
            )
            datasets.append(dataset)

        combined_dataset = concatenate_datasets(datasets)
        combined_dataset.to_json(output_dataset_path, orient="records", lines=True)
        return output_dataset_path

    def estimate_max_costs(self, dataset_size: int) -> dict[str, Any]:
        price_per_1m_tokens = self.PRICING_PER_1M_TOKENS[self.model]
        max_token_count = self.max_prompt_token_count * dataset_size
        regular_price = price_per_1m_tokens * max_token_count / 1_000_000
        batch_api_price = regular_price / 2  # typically half
        num_of_days = math.ceil(max_token_count / self.batch_limit_tpd)
        return {
            "costs": f"${regular_price:.3f}",
            "costs (batch)": f"${batch_api_price:.3f}",
            "days to complete (batch)": num_of_days,
        }


def submit_chat_completion_job(client: OpenAI, batch_job_file: Path) -> str:
    with open(batch_job_file, "rb") as f:
        file_response = client.files.create(file=f, purpose="batch")
        file_id = file_response.id
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Batch for {batch_job_file.name}"},
    )
    return batch_response.id


def wait_for_chat_completion_job(
    client: OpenAI,
    batch_job_id: str,
    output_file_path: Path,
    sleep_time_seconds: float = 60,
) -> tuple[bool, Optional[str], Optional[list[dict[str, str]]]]:
    if output_file_path.exists():
        return True, None, None

    errors = None
    while True:
        batch_job = client.batches.retrieve(batch_job_id)
        status = batch_job.status
        if status == "completed":
            output_content = client.files.content(batch_job.output_file_id).text
            with open(output_file_path, "w") as f:
                f.write(output_content)
            break
        elif status == "failed":
            errors = json.loads(batch_job.errors.to_json())["data"]
            break
        elif status in ["cancelling", "cancelled", "expired"]:
            break
        # make sure we are not skipping any status
        assert status in ["validating", "finalizing", "in_progress"]
        time.sleep(sleep_time_seconds)

    return output_file_path.exists(), status, errors


def num_tokens_from_messages(messages: list[dict[str, str]], model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages.
    Lifted from https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print(
            "Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18."
        )
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print(
            "Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06."
        )
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
