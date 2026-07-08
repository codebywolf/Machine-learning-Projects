import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_classification_matrix(y_true, y_pred, y_prob=None):
    logger.info("Calculating baseline performance metrics...")

    # Ensure inputs are standard 1D arrays to avoid format errors
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob).ravel()
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC score: {e}")
            metrics['roc_auc'] = 0.0

    return metrics

def generate_confusion_matrix_plot(y_true, y_pred, output_image_path:str):
    logger.info('Generating normalized confusion matrix...')
    cm = confusion_matrix(y_true, y_pred)

    # Normalizing by raw
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting normalized confusion matrics
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()

    if output_image_path:
        plt.savefig(output_image_path, dpi=300)
        logger.info(f"Confusion matrix plot successfully saved to {output_image_path}")

    plt.close()
