import sys

from ollama import ChatResponse, Client
from transformers import AutoTokenizer

from .chat_completer import LLMChatCompleter, LLMChatCompleterResponse


class OllamaChatCompleter(LLMChatCompleter):
    def __init__(
        self,
        model: str,
        tokenizer: str,
        host: str = "http://localhost:11434",
        headers: dict[str, str] | None = None,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.temperature = temperature
        self.client = Client(host=host, headers=headers or {})

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
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                max_length=sys.maxsize,  # to suppress a warning
            )
        return len(self.encode(message))

    def encode(self, message: str) -> list[int]:
        return self.tokenizer(
            [message],
            max_length=sys.maxsize,  # to suppress a warning
        )[
            "input_ids"
        ][0]

    def decode(self, tokens: list[int]) -> list[int]:
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return response
