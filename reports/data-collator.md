# Data Collator Experiments

In this notebook, we'll explore how to construct batches out of processed `Commit Chronicle` dataset during the training/validation setting for a encoder-decoder style architecture.

**Make sure to run `commit-chronicle-dataset.ipynb` before using this notebook.**

The logic laid out in this notebook is implemented in `DataCollatorTrain`.


```python
import rootutils
import torch
import torch.utils.data
from datasets import load_from_disk
```


```python
ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)
OUTPUT_DIR = ROOT / "data/playground"
```


```python
from src.data.types import SingleExample
```


```python
dataset_ = load_from_disk(OUTPUT_DIR / "02-processed-validation")
dataset_.select(range(10)).to_pandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>msg_input_ids</th>
      <th>diff_input_ids</th>
      <th>language</th>
      <th>repo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>770513</td>
      <td>[986, 981, 358, 4614, 326, 3434, 729, 1867, 60...</td>
      <td>[2704, 585, 2713, 19, 1425, 19, 1425, 18, 3240...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>1</th>
      <td>770513</td>
      <td>[986, 1390, 364, 5023, 326, 2521, 471, 19171, ...</td>
      <td>[2704, 585, 276, 517, 2656, 18, 3240, 203, 15,...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>2</th>
      <td>770513</td>
      <td>[986, 279, 24778, 716, 22991, 4167, 326, 4422,...</td>
      <td>[2704, 585, 10746, 958, 18, 1264, 203, 15, 7, ...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>3</th>
      <td>770513</td>
      <td>[986, 3270, 358, 1086, 707]</td>
      <td>[7236, 19, 2910, 18, 3240, 203, 30989, 300, 26...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>4</th>
      <td>770513</td>
      <td>[5726, 2172, 7153, 16, 29288, 364, 946, 310, 4...</td>
      <td>[6949, 958, 18, 1264, 203, 30989, 300, 5558, 1...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>5</th>
      <td>770513</td>
      <td>[6464, 4677, 461, 3152, 6810]</td>
      <td>[7236, 19, 1425, 19, 1425, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>6</th>
      <td>770513</td>
      <td>[986, 1122, 4409, 18, 203, 1986, 4409, 1914, 7...</td>
      <td>[7236, 19, 2910, 18, 3240, 203, 30989, 300, 27...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>7</th>
      <td>770513</td>
      <td>[7505, 279, 7934, 1625, 777, 2743, 434, 279, 1...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>8</th>
      <td>770513</td>
      <td>[5058, 854, 2037, 777, 4203, 16, 309, 1915, 45...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>9</th>
      <td>770513</td>
      <td>[8585, 1904, 7153, 16, 2037, 777, 2743, 854, 4...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
  </tbody>
</table>
</div>




```python
class HumbleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> SingleExample:
        row = self.dataset[index]
        return SingleExample(
            diff_input_ids=row["diff_input_ids"],
            msg_input_ids=row["msg_input_ids"],
            history_input_ids=[],  # ignored in this notebook. don't worry about it. trust me :)
            pos_in_file=-1,  # ignored in this notebook.
        )


data = HumbleDataset(dataset_)
```

Let's load the tokenizers we used in `commit-chronicle-dataset.ipynb`. We are done tokenizing but we need to known the identities of some special tokens, i.e. beginning of sentence (BOS) token, end of sentence (EOS) token, and the pad token.


```python
from copy import deepcopy

from transformers import AutoTokenizer

from src.data.components.tokenization import add_special_tokens

msg_tokenizer_ = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
msg_tokenizer_ = add_special_tokens(msg_tokenizer_, None)
diff_tokenizer_ = deepcopy(msg_tokenizer_)
```

# Encoder Input Processing

Here we assume input to the encoder is the git diff, `diff_input_ids` attribute of `SingleExample`. It can also be history of all git diffs, but we don't use it here.


```python
def process_encoder_inputs(
    input_ids_batch: list[list[int]],
    encoder_context_max_len: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
):
    """
    This helper method processes either diffs or messages as encoder input.

    It truncates the inputs to the maximum allowed length.

    It also adds all required special tokens: format is [BOS] input [EOS].

    Finally, it is responsible for padding to maximum length in batch and conversion to torch.Tensor.

    Args:
        input_ids_batch: A list of tokenized examples from the current batch.
        encoder_context_max_len: The maximum length of the encoder context.
        bos_token_id: The value of the beginning of sequence (BOS) token.
        eos_token_id: The value of the end of sequence (EOS) token.
        pad_token_id: The value of the padding token (PAD) token.

    Returns:
        input_ids for encoder, attention_mask for encoder
    """

    # add BOS and EOS tokens to each example whilst making sure max length of resulting token list is encoder_context_max_len
    input_ids_batch = [
        [bos_token_id] + example[: encoder_context_max_len - 2] + [eos_token_id]
        for example in input_ids_batch
    ]
    inputs_tensors = [torch.tensor(ids, dtype=torch.int64) for ids in input_ids_batch]

    # pad tensors to max length in batch
    inputs_max_len = max(len(tensor) for tensor in input_ids_batch)
    inputs_tensors = [
        _pad_tensor(
            tensor,
            pad_len=inputs_max_len - tensor.numel(),
            value=pad_token_id,
            left=False,
        )
        for tensor in inputs_tensors
    ]

    masks_tensors = [torch.ones_like(ids) for ids in inputs_tensors]
    masks_tensors = [
        _pad_tensor(
            tensor,
            pad_len=inputs_max_len - tensor.numel(),
            value=0,
            left=False,
        )
        for tensor in masks_tensors
    ]
    return torch.stack(inputs_tensors), torch.stack(masks_tensors)


def _pad_tensor(input_tensor: torch.Tensor, pad_len: int, value: int, left: bool) -> torch.Tensor:
    return torch.nn.functional.pad(
        input_tensor,
        pad=[pad_len, 0] if left else [0, pad_len],
        mode="constant",
        value=value,
    )
```

Let's try it out with a batch size of 2


```python
examples_ = [data[0], data[1]]
git_diff_inputs_ = [example.diff_input_ids for example in examples_]
encoder_input_ids_, encoder_attention_mask_ = process_encoder_inputs(
    input_ids_batch=git_diff_inputs_,
    encoder_context_max_len=512,  # this is a hyperparameter
    bos_token_id=diff_tokenizer_.bos_token_id,
    eos_token_id=diff_tokenizer_.eos_token_id,
    pad_token_id=diff_tokenizer_.pad_token_id,
)
encoder_input_ids_.shape, encoder_attention_mask_.shape
```




    (torch.Size([2, 512]), torch.Size([2, 512]))



That's it. The output data forms input to our encoder.

# Decoder Input


```python
from typing import Literal, Optional


def _process_decoder_input(
    examples: list[SingleExample],
    msg_bos_token_id: int,
    msg_eos_token_id: int,
    msg_pad_token_id: int,
    decoder_context_max_len,
    shift_labels: bool,
    decoder_start_token_id: Optional[int] = None,
    # ignore these options
    encoder_input_type: Literal["diff", "history"] = "diff",
    with_history: bool = False,
):
    """
    Prepares decoder input for train/validation:
      * aggregates messages from history when configured accordingly
      * concatenates history with current message
      * constructs labels
      * pads, converts to tensors

    Args:
        examples: A list of inputs for current batch.

    Returns:
        Tuple of three tensors: input ids, attention masks, labels.
    """
    message_inputs: list[list[int]] = [example.msg_input_ids for example in examples]
    history_inputs: list[list[list[int]]] = [example.history_input_ids for example in examples]

    all_msg_ids: list[torch.Tensor] = []
    all_msg_masks: list[torch.Tensor] = []
    all_msg_labels: list[torch.Tensor] = []

    for message_ids, history_ids in zip(message_inputs, history_inputs):
        message_ids = message_ids[: decoder_context_max_len - 2]

        cur_history_ids = []
        cur_history_labels = []

        # if encoder_input_type != "history" and with_history:
        #     cur_history_ids = _get_history(
        #         cur_len=len(message_ids) + 2,
        #         history_ids=history_ids,
        #     )
        #     cur_history_labels = [
        #         [-100 for _ in message] for message in cur_history_ids
        #     ]

        cur_ids = [[msg_bos_token_id]] + cur_history_ids + [message_ids] + [[msg_eos_token_id]]
        cur_labels = (
            [[msg_bos_token_id]] + cur_history_labels + [message_ids] + [[msg_eos_token_id]]
        )

        if shift_labels:
            cur_ids, cur_labels = _shift_for_encoder_decoder(
                cur_ids,
                cur_labels,
                msg_bos_token_id=msg_bos_token_id,
                decoder_start_token_id=decoder_start_token_id,
            )

        cur_ids_tensor = torch.tensor(
            [ex for sublist in cur_ids for ex in sublist], dtype=torch.int64
        )
        cur_labels_tensor = torch.tensor(
            [ex for sublist in cur_labels for ex in sublist], dtype=torch.int64
        )
        cur_mask_tensor = torch.ones_like(cur_ids_tensor)

        all_msg_ids.append(cur_ids_tensor)
        all_msg_masks.append(cur_mask_tensor)
        all_msg_labels.append(cur_labels_tensor)

    msg_max_len = max(len(tensor) for tensor in all_msg_ids)
    all_msg_ids = [
        _pad_tensor(
            tensor,
            pad_len=msg_max_len - tensor.numel(),
            value=msg_pad_token_id,
            left=False,
        )
        for tensor in all_msg_ids
    ]
    all_msg_masks = [
        _pad_tensor(
            tensor,
            pad_len=msg_max_len - tensor.numel(),
            value=0,
            left=False,
        )
        for tensor in all_msg_masks
    ]
    all_msg_labels = [
        _pad_tensor(
            tensor,
            pad_len=msg_max_len - tensor.numel(),
            value=-100,
            left=False,
        )
        for tensor in all_msg_labels
    ]

    return (
        torch.stack(all_msg_ids),
        torch.stack(all_msg_masks),
        torch.stack(all_msg_labels),
    )


def _shift_for_encoder_decoder(
    ids: list[list[int]],
    labels: list[list[int]],
    msg_bos_token_id: int,
    decoder_start_token_id: Optional[int] = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """This method mimics transformers logic of ids and labels for EncoderDecoderModel
    (or T5ForConditionalGeneration).

    Starting from transformers v4.12, loss is now calculated in EncoderDecoderModel, not in decoder class.
    Also, decoder input ids are created automatically based on labels: labels are shifted and -100 is replaced
    with pad token. In our case, history ids are masked -100 in labels, but they are still
    meaningful ids. Therefore, we can't use the default approach.
    """
    if decoder_start_token_id is None:
        ids = [[msg_bos_token_id]] + ids[:-1]
    else:
        ids = [[decoder_start_token_id]] + ids[:-1]
    return ids, labels
```

Trying it out


```python
decoder_input_ids_, decoder_attention_mask_, labels_ = _process_decoder_input(
    examples=examples_,
    msg_bos_token_id=msg_tokenizer_.bos_token_id,
    msg_eos_token_id=msg_tokenizer_.eos_token_id,
    msg_pad_token_id=msg_tokenizer_.pad_token_id,
    decoder_context_max_len=512,  # this is a hyperparam for the model
    shift_labels=True,
)
decoder_input_ids_.shape, decoder_attention_mask_.shape, labels_.shape
```




    (torch.Size([2, 26]), torch.Size([2, 26]), torch.Size([2, 26]))



# Model Testing


```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
```


```python
outputs = model(
    input_ids=encoder_input_ids_,
    attention_mask=encoder_attention_mask_,
    decoder_input_ids=decoder_input_ids_,
    decoder_attention_mask=decoder_attention_mask_,
    labels=labels_,
)
[attr for attr in dir(outputs) if not attr.startswith("_")]
```

    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
    




    ['clear',
     'copy',
     'cross_attentions',
     'decoder_attentions',
     'decoder_hidden_states',
     'encoder_attentions',
     'encoder_hidden_states',
     'encoder_last_hidden_state',
     'fromkeys',
     'get',
     'items',
     'keys',
     'logits',
     'loss',
     'move_to_end',
     'past_key_values',
     'pop',
     'popitem',
     'setdefault',
     'to_tuple',
     'update',
     'values']




```python
outputs.loss
```




    tensor(13.9106, grad_fn=<NllLossBackward0>)




```python
# let's overfit
from tqdm import tqdm

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.1)

for i in range(100):
    optimizer.zero_grad()
    outputs = model(
        input_ids=encoder_input_ids_,
        attention_mask=encoder_attention_mask_,
        decoder_input_ids=decoder_input_ids_,
        decoder_attention_mask=decoder_attention_mask_,
        labels=labels_,
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch: {i:03d} Loss:{loss:.4f}")
```

    Epoch: 000 Loss:13.4673
    Epoch: 001 Loss:13.8893
    Epoch: 002 Loss:14.6309
    Epoch: 003 Loss:13.5598
    Epoch: 004 Loss:14.4649
    Epoch: 005 Loss:13.3989
    Epoch: 006 Loss:13.0234
    Epoch: 007 Loss:13.4223
    Epoch: 008 Loss:13.6826
    Epoch: 009 Loss:12.9208
    Epoch: 010 Loss:13.5239
    Epoch: 011 Loss:13.7620
    Epoch: 012 Loss:12.8485
    Epoch: 013 Loss:13.7733
    Epoch: 014 Loss:13.8697
    Epoch: 015 Loss:13.8741
    Epoch: 016 Loss:14.5653
    Epoch: 017 Loss:13.6485
    Epoch: 018 Loss:13.3812
    Epoch: 019 Loss:14.2164
    Epoch: 020 Loss:13.7952
    Epoch: 021 Loss:14.1360
    Epoch: 022 Loss:13.0741
    Epoch: 023 Loss:14.1174
    Epoch: 024 Loss:13.4539
    Epoch: 025 Loss:15.0065
    Epoch: 026 Loss:13.3655
    Epoch: 027 Loss:13.0178
    Epoch: 028 Loss:12.2077
    Epoch: 029 Loss:14.6857
    Epoch: 030 Loss:13.8917
    Epoch: 031 Loss:12.7636
    Epoch: 032 Loss:13.2466
    Epoch: 033 Loss:14.1721
    Epoch: 034 Loss:16.0520
    Epoch: 035 Loss:14.1160
    Epoch: 036 Loss:13.2025
    Epoch: 037 Loss:14.2383
    Epoch: 038 Loss:13.0495
    Epoch: 039 Loss:14.4571
    Epoch: 040 Loss:13.1915
    Epoch: 041 Loss:12.9976
    Epoch: 042 Loss:13.2691
    Epoch: 043 Loss:14.3251
    Epoch: 044 Loss:14.3469
    Epoch: 045 Loss:14.2013
    Epoch: 046 Loss:13.8181
    Epoch: 047 Loss:14.4091
    Epoch: 048 Loss:14.2068
    Epoch: 049 Loss:14.4967
    Epoch: 050 Loss:13.3913
    Epoch: 051 Loss:16.1312
    Epoch: 052 Loss:13.7539
    Epoch: 053 Loss:14.4688
    Epoch: 054 Loss:15.0127
    Epoch: 055 Loss:12.9980
    Epoch: 056 Loss:13.2712
    Epoch: 057 Loss:13.5811
    Epoch: 058 Loss:13.7861
    Epoch: 059 Loss:13.3325
    Epoch: 060 Loss:12.7937
    Epoch: 061 Loss:13.6691
    Epoch: 062 Loss:14.8429
    Epoch: 063 Loss:13.4066
    Epoch: 064 Loss:13.0776
    Epoch: 065 Loss:14.0721
    Epoch: 066 Loss:12.8465
    Epoch: 067 Loss:12.8599
    Epoch: 068 Loss:14.3283
    Epoch: 069 Loss:13.7042
    Epoch: 070 Loss:14.2878
    Epoch: 071 Loss:12.2032
    Epoch: 072 Loss:13.6322
    Epoch: 073 Loss:12.2535
    Epoch: 074 Loss:13.7174
    Epoch: 075 Loss:13.3022
    Epoch: 076 Loss:13.5978
    Epoch: 077 Loss:13.2317
    Epoch: 078 Loss:13.3742
    Epoch: 079 Loss:13.2306
    Epoch: 080 Loss:13.9516
    Epoch: 081 Loss:12.4263
    Epoch: 082 Loss:13.1299
    Epoch: 083 Loss:13.5563
    Epoch: 084 Loss:14.1990
    Epoch: 085 Loss:14.8061
    Epoch: 086 Loss:13.7511
    Epoch: 087 Loss:13.3625
    Epoch: 088 Loss:12.6832
    Epoch: 089 Loss:14.4581
    Epoch: 090 Loss:12.3702
    Epoch: 091 Loss:13.2385
    Epoch: 092 Loss:13.1922
    Epoch: 093 Loss:12.3057
    Epoch: 094 Loss:13.8241
    Epoch: 095 Loss:12.4620
    Epoch: 096 Loss:13.8702
    Epoch: 097 Loss:13.0230
    Epoch: 098 Loss:12.6946
    Epoch: 099 Loss:13.7120
    

I was expecting the model loss to reduce smoothly but that didn't happen. Hmmmmm...
