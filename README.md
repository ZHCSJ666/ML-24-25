<!--suppress HtmlDeprecatedAttribute -->
<div align="center">

# Git Commit Message Generation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This project aims to use machine learning and natural language processing techniques to automatically generate high-quality Git commit messages based on code changes (git diff).

## Installation

#### Conda

```bash
# clone project
git https://github.com/ZHCSJ666/ML-24-25
cd ML-24-25

# create conda environment and install dependencies
conda env create --name cmg -f environment.yaml

# activate conda environment
conda activate cmg
```

If for some reason, imports are not working well for you, you can install the project as a package.

```bash
# run command from project root directory
pip install -e . --config-settings editable_mode=compat
```

## **Data Simplification**

The main script for dataset simplification is **[src/simplify_data.py](src/simplify_data.py)**, with its corresponding configuration file located at **[configs/simplify-data.yaml](configs/simplify-data.yaml)**.

### **Modes of Operation**

The script supports three modes:

1. **`prompt-testing`**  
   - Runs a single prompt on an example and prints the result.
  
2. **`eager`**  
   - Processes the entire dataset by making an API request for each example.
  
3. **`batch`** *(Supported only by GPT)*  
   - Uses batch requests for processing, typically executing within a 24-hour window.

### **Supported LLM APIs for Chat Completion**

The script supports two categories of LLM-based chat completion APIs:

1. **GPT**  
   - Implemented in **[GPT Chat Completer](src/utils/chat_completion/gpt_chat_completer.py)**  
   - This interfaces with OpenAI's API or an API (like [DeepSeek](https://api-docs.deepseek.com/)) that's compatible
   with OpenAI's API.
  
2. **Ollama**  
   - Implemented in **[Ollama Chat Completer](src/utils/chat_completion/ollama_chat_completer.py)**  
   - Requires installing and running [Ollama](https://ollama.com/) locally or using a cloud-based Ollama instance.
   - Pull desired Ollama model first before running `src/simplify_data.py` e.g.  `ollama pull qwen2.5-coder:7b`
For examples of currently configured LLMs, refer to **[Chat Completer Config](configs/chat_completer)**.

### **Usage Examples**

1. [qwen2.5-coder-7b](configs/chat_completer/qwen2.5-coder-7b.yaml)
```bash
# Example running a prompt test
ollama pull qwen2.5-coder:7b
python src/simplify_data.py chat_completer=qwen2.5-coder-7b mode=prompt-testing split=test

# Example debugging an eager mode run
ollama pull qwen2.5-coder:7b # this should if the model hasn't been previously pulled or has been deleted
python src/simplify_data.py chat_completer=qwen2.5-coder-7b mode=eager debug_run=True

# Example running an eager mode run
ollama pull qwen2.5-coder:7b
python src/simplify_data.py chat_completer=qwen2.5-coder-7b mode=eager debug_run=False
```
2. [GPT-4o-mini](configs/chat_completer/gpt-4o-mini.yaml)

Using [OpenAI's API](https://platform.openai.com/docs/overview) requires setting the `OPENAI_API_KEY` environment variable.
```bash
# Example running a prompt test
python src/simplify_data.py chat_completer=gpt-4o-mini mode=prompt-testing split=test
```

3. [DeepSeek-v3](configs/chat_completer/deepseek-v3.yaml)

Using [DeepSeek's API](https://api-docs.deepseek.com/) requires setting the `DEEPSEEK_API_KEY` environment variable.
```bash
# Example running a prompt test
python src/simplify_data.py chat_completer=deepseek-v3 mode=prompt-testing split=test
```
## Git Commit Message Generation Experiments

The task of git commit message can be posed in a number of ways. So far we've experimented with two approaches
1. Sequence-to-sequence task
2. Causal language modeling

### Sequence-to-Sequence

For transformer based models, during training, input to the encoder is git diffs and input to the decoder is the ground
truth commit message.

Examples
1. t5-efficient-tiny
```shell
python src/train.py experiment=t5-efficient-tiny logger=tensorboard
```

### Causal Language Modeling

Decoder-only models are popular candidates for casual language modeling. Here input diff and ground truth commit message 
concatenated with a special (separation token) in between. During testing, the git diff is appended at its end by the
separation token.

Examples
1. pythia-14m
```shell
# (debug) fast dev run
python src/train.py experiment=pythia logger=tensorboard +trainer.overfit_batches=3 trainer.max_epochs=50

# to train on full dataset
python src/train.py experiment=pythia logger=tensorboard
```

## Important Questions and Answers


##### 1. What NN architectures did we experiment with?

- T5-efficient with config modified reducing number of parameters to 7.3M (t5-efficient-extra-tiny)
- Decoder-only model (pythia)

##### 2. What is the size of the dataset used for training?

The number of tokens trained on is at least 20 times the number of parameters in the t5-efficient-extra-tiny model.
The rationale behind the number of tokens used is explained [here](https://arxiv.org/abs/2406.12907).
See [Chinchilla data-optimal scaling laws: In plain English](https://lifearchitect.ai/chinchilla/) for explanation of the paper in plain English.

##### 3. What does the dataset looks like?

It's a simplified version of the original Commit Chronicle dataset from Jetbrains. It uses only Golang, 10% of changes with MODIFY changes types,
and all other change types.

The commit message was simplified using a local version of qwen2.5-coder-7b with the help of Ollama. Simplifying the selected fraction of
the original dataset took around **12 hours**.

##### 4. What tokenizer was used?

The tokenizer used for training the t5 model is the T5-efficient tokenizer from Hugging Face.

TODO: Need to go into more detail about how the tokenizer works.

##### 5. How long did training take? How many epochs?


## How to run

Training examples with [t5-efficient-tiny](configs/experiment/t5-efficient-tiny.yaml) experiment configuration

```bash
# train with flan-t5 model on Commit Chronicle dataset
python src/train.py experiment=t5-efficient-tiny logger=tensorboard

# (debug) overfit on subset of training data
python src/train.py experiment=t5-efficient-tiny logger=tensorboard +trainer.overfit_batches=3 trainer.max_epochs=50

# (debug) fast dev run
python src/train.py experiment=t5-efficient-tiny logger=tensorboard +trainer.fast_dev_run=True trainer=cpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
