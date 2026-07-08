import os
import pandas as pd
import numpy as np
import torch
import logging
from transformers import (
        RobertaTokenizer,
        RobertaForSequenceClassification,
        Trainer,
        TrainingArguments
)

from src.utils import load_config
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.data_loader import FakeNewsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@staticmethod
def prepare_transform_pipeline():
    config = load_config(config_path= '../configs/roberta_config.yaml')
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['logging_dir'], exist_ok=True)

    logger.info(f"Initializing Tokenizer: {config['model']['model_name_or_path']}")
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['model_name_or_path'])
    return config, tokenizer

def compute_metrics(eval_pred):
    # Computes precision, recall, f1, and accuracy for sequence classification.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

if __name__ == "__main__":
    # Initialize Pipeline Configs and Tokenizer
    config, tokenizer = prepare_transform_pipeline()

    # Ingest the clean CSV data frames directly
    logger.info("Ingesting clean datasets for Transformer Optimization...")
    train_df = pd.read_csv("../data/processed/processed_train_df.csv").dropna(subset=['text']).reset_index(drop=True)
    val_df = pd.read_csv("../data/processed/processed_val_df.csv").dropna(subset=['text']).reset_index(drop=True)

    # Format into custom PyTorch Dataset wrappers
    logger.info("Encoding text data fields into PyTorch tensor structures...")
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=config['model']['max_length']
    )

    val_dataset = FakeNewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=config['model']['max_length']
    )

    # Initialize Pre-trained RoBERTa Architecture
    logger.info("Downloading pre-trained RoBERTa sequence classification weights...")
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['model_name_or_path'],
        num_labels=config['model']['num_labels']
    )

    # Build Training Arguments from YAML parameters
    t_args = config['training_args']
    training_arguments = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        logging_dir=config['paths']['logging_dir'],
        num_train_epochs=t_args['num_train_epochs'],
        per_device_train_batch_size=t_args['per_device_train_batch_size'],
        per_device_eval_batch_size=t_args['per_device_eval_batch_size'],
        warmup_steps=t_args['warmup_steps'],
        weight_decay=t_args['weight_decay'],
        learning_rate=float(t_args['learning_rate']),
        eval_strategy=t_args['evaluation_strategy'],
        save_strategy=t_args['save_strategy'],
        load_best_model_at_end=t_args['load_best_model_at_end'],
        metric_for_best_model=t_args['metric_for_best_model'],
        fp16=t_args['fp16'],
        logging_steps=100
    )

    # Instantiate Hugging Face Trainer Instance
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 7. Execute Fine-Tuning Optimization Execution Run
    logger.info("Launching Transformer training loop on active hardware acceleration device...")
    trainer.train()