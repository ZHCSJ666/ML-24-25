import os

# Set environment variables for offline mode and cache directories
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/users/liamding/data/LLAVA-2"
os.environ["HF_HOME"] = "/mnt/data/users/liamding/data/LLAVA-2"

# Add NLTK data path
import nltk
nltk.data.path.append('/mnt/data/users/liamding/data/LLAVA-2')

import sacrebleu
from bert_score import score
import json
import sys
from nltk.translate import meteor_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the local model path
bert_name = "/mnt/data/users/liamding/data/LLAVA-2/bert-base-chinese"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_name, local_files_only=True)
logger.info(f"Tokenizer successfully loaded from {bert_name}.")

# Load the model configuration to get the number of layers
config = AutoConfig.from_pretrained(bert_name, local_files_only=True)
num_layers = config.num_hidden_layers

# Load the model
model = AutoModel.from_pretrained(bert_name, local_files_only=True)
logger.info(f"Model successfully loaded from {bert_name}.")

def bleu_score(predict, answer):
    """
    Calculate BLEU score
    """
    bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="flores101")
    return bleu.score

def chrf_score(predict, answer):
    """
    Calculate CHRF score
    """
    chrf = sacrebleu.corpus_chrf(predict, answer)
    return chrf.score

def ter_score(predict, answer):
    """
    Calculate TER score
    """
    ter = sacrebleu.corpus_ter(predict, answer, asian_support=True, normalized=True, no_punct=True)
    return ter.score

def bertscore(predict, answer):
    """
    Calculate BERTScore
    """
    P, R, F1 = score(
        cands=predict,
        refs=answer,
        model_type=bert_name,  # Use the local model path
        num_layers=num_layers,
        lang='zh',
        verbose=True,
        idf=False,
        rescale_with_baseline=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return torch.mean(P).item(), torch.mean(R).item(), torch.mean(F1).item()

def meteor(predict, answer, type):
    """
    Calculate METEOR score
    """
    all_meteor = []
    for i in range(len(predict)):
        meteor = meteor_score.meteor_score(
            [tokenizer.tokenize(answer[i])],
            tokenizer.tokenize(predict[i]),
        )
        all_meteor.append(meteor)
    if type == "total":
        return sum(all_meteor) / len(all_meteor)
    else:
        return all_meteor[0]

def extract(data, target):
    """
    Extract prediction and reference translations from data
    """
    predicts = []
    answers = []
    logger.info("Extracting data...")
    for item in tqdm(data):
        id = item["id"]
        pred = item['target'].strip()
        ans = target[id].strip()
        predicts.append(pred)
        answers.append(ans)
    logger.info("Data extraction completed.")
    return predicts, answers

def cal_total_metrics(predicts, answers):
    """
    Calculate overall evaluation metrics
    """
    logger.info("Calculating overall metrics...")
    bs = bleu_score(predicts, [answers])
    logger.info(f"BLEU: {bs}")
    cs = chrf_score(predicts, [answers])
    logger.info(f"CHRF: {cs}")
    ts = ter_score(predicts, [answers])
    logger.info(f"TER: {ts}")
    p, r, f1 = bertscore(predicts, answers)
    logger.info(f"BERT-P: {p}, BERT-R: {r}, BERT-F1: {f1}")
    m = meteor(predicts, answers, "total")
    logger.info(f"METEOR: {m}")

    print("BLEU:", bs)
    print("CHRF:", cs)
    print("TER:", ts)
    print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
    print("METEOR:", m)

    res = [{"BLEU": bs, "CHRF": cs, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m}]
    df = pd.DataFrame(res)
    df.to_csv(data_path.with_name(data_path.stem + "_total.csv"), index=False, encoding='utf-8-sig')
    logger.info(f"Overall metrics saved to {data_path.with_name(data_path.stem + '_total.csv')}")

def cal_each_metrics(predicts, answers):
    """
    Calculate evaluation metrics for each sample
    """
    logger.info("Calculating metrics for each sample...")
    all_result = []
    for i in tqdm(range(len(predicts))):
        ans = answers[i]
        pred = predicts[i]
        bs = bleu_score([pred], [[ans]])
        cs = chrf_score([pred], [[ans]])
        ts = ter_score([pred], [[ans]])
        p, r, f1 = bertscore([pred], [ans])
        m = meteor([pred], [ans], "each")
        all_result.append({
            "reference": ans,
            "predicts": pred,
            "BLEU": bs,
            "CHRF": cs,
            "TER": ts,
            "BERT-P": p,
            "BERT-R": r,
            "BERT-F1": f1,
            "METEOR": m
        })
    df = pd.DataFrame(all_result)
    df.to_csv(data_path.with_name(data_path.stem + "_each.csv"), index=False, encoding='utf-8-sig')
    logger.info(f"Metrics for each sample saved to {data_path.with_name(data_path.stem + '_each.csv')}")

if __name__ == "__main__":
    # Paths to input data and reference translations
    data_file = "/mnt/data/users/liamding/data/3AM/3AM/original.json"
    data_path = Path(data_file)
    target_file = "/mnt/data/users/liamding/data/3AM/3AM/data/test.zh"

    logger.info("Loading reference translations...")
    with open(target_file, "r", encoding="utf-8") as f:
        target = f.readlines()
    logger.info("Loading model-generated translations...")
    data = json.load(open(data_file, "r", encoding="utf-8"))

    logger.info("Extracting predictions and references...")
    predicts, answers = extract(data, target)

    logger.info("Calculating metrics for each sample...")
    print("Metrics for each sample:")
    cal_each_metrics(predicts, answers)

    logger.info("Calculating overall metrics...")
    print("Overall metrics:")
    cal_total_metrics(predicts, answers)

    logger.info("Evaluation completed.")
