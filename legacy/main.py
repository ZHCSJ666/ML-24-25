# main.py

import argparse
import yaml
from data.collect_data import collect_data
from data.preprocess_data import preprocess_data
from models.train import train_model
from models.evaluate import evaluate_model

def main(config_path):
    # Load configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Data collection
    collect_data(config)

    # Data preprocessing
    preprocess_data(config)

    # Model training
    train_model(config)

    # Model evaluation
    evaluate_model(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git Commit Message Generation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)
