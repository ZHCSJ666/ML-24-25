FROM mambaorg/micromamba:2.0-cuda12.4.1-ubuntu22.04

WORKDIR /opt/project

# setup python environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml .
RUN micromamba install libcufile=1.11.1.6 --yes
RUN micromamba install --yes --name base -f environment.yaml
RUN micromamba clean --all --yes

# test python environment
RUN /opt/conda/bin/python -c "import torch;print(torch.version.cuda);"

# fetch dataset (and other huggingface assets)
# this makes our image huge (but it circumvents the China firewall issue)
RUN /opt/conda/bin/python -c "from datasets import load_dataset;load_dataset('JetBrains-Research/commit-chronicle', 'default');"
RUN /opt/conda/bin/python -c "from transformers import AutoTokenizer;AutoTokenizer.from_pretrained('Salesforce/codet5-small');"

# copy src code
COPY .project-root ./
COPY src/ src/
COPY configs configs/


ENTRYPOINT ["top", "-b"]