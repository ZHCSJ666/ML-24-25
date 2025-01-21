import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
from src.data.types import Batch

# https://stackoverflow.com/a/53014308/7121776
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class DecoderWrapper(nn.Module):
    """This class serves as a `AutoModelForCausalLM` wrapper for commit message completion task."""

    def __init__(self, config, tokenizer):
        super().__init__()

        config.vocab_size = tokenizer.vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_config(config)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, batch: Batch):
        output: CausalLMOutputWithPast = self.model(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            labels=batch.labels,
        )
        return {"logits": output.logits, "loss": output.loss}

    def generate(self, batch: Batch, **generation_kwargs):
        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            **generation_kwargs,
        )
