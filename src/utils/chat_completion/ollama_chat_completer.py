import sys

from ollama import ChatResponse, Client
from transformers import AutoTokenizer

from .chat_completer import LLMChatCompleter, LLMChatCompleterResponse


class OllamaChatCompleter(LLMChatCompleter):
    """
    A chat completion handler for Ollama-based models.

    This class interacts with an Ollama server to generate chat completions
    using a specified LLM model. It supports message encoding, decoding,
    and token counting using Hugging Face's tokenizer.

    Attributes:
        model (str): The Ollama model name (e.g., "qwen2.5-coder:14b").
        tokenizer (AutoTokenizer): The Hugging Face tokenizer for the model.
        host (str): The Ollama server address (default: "http://localhost:11434").
        headers (dict[str, str]): Custom HTTP headers for API requests.
        temperature (float): The temperature setting for the model's response generation.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        host: str = "http://localhost:11434",
        headers: dict[str, str] | None = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initializes the OllamaChatCompleter.

        Args:
            model: The Ollama model name (e.g., "qwen2.5-coder:14b").
            tokenizer: The full Hugging Face tokenizer name used by the model
                (e.g., for "qwen2.5-coder:14b", the tokenizer is "Qwen/Qwen2.5-Coder-14B-Instruct").
            host: The address of the Ollama server. Defaults to "http://localhost:11434".
            headers: Custom client headers for API requests. Defaults to None.
            temperature: Controls randomness in the model’s responses (higher values = more randomness).
                Defaults to 0.7.
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.temperature = temperature
        self.host = host
        self.headers = headers or {}

    @property
    def client(self):
        return Client(host=self.host, headers=self.headers)

    def complete_chat(self, messages: list[dict[str, str]]) -> LLMChatCompleterResponse:
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_thread": 16},
        )
        return LLMChatCompleterResponse(
            content=response.message.content,
            prompt_token_count=response.prompt_eval_count,
            response_token_count=response.eval_count,
        )

    def count_tokens(self, message: str | list[dict[str, str]]) -> int:
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

    def encode(self, message: str) -> list[int]:
        return self.tokenizer(
            [message],
            truncation=False,
            # to suppress a warning
            max_length=sys.maxsize,
        )["input_ids"][0]

    def decode(self, tokens: list[int]) -> list[int]:
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return response
