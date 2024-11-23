"""The code in this file mimic behavior of some Huggingface transformers optimizer and scheduling logic"""

import torch.nn as nn
from torch.nn import LayerNorm
from torch.optim.adamw import AdamW
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """Creates an AdamW optimizer based on the implementation from Huggingface Transformers to skip LayerNorm in weight decay.
    See https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L1019
    """
    decay_parameters = get_parameter_names(model, [LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
            "name": "decayed",
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "name": "no_decay",
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)
