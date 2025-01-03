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

## Experiments

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
