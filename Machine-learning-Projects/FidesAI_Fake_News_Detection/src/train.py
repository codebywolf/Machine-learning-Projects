import os
import joblib
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from data_loader import prepare_raw_isot, load_and_merge_datasets, create_stratified_splits
from preprocess import TextCleaner
from evaluate import compute_classification_matrix, generate_confusion_matrix_plot
from features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_baseline_pipeline(
        welfake_path: str,
        isot_true_path: str = None,
        isot_fake_path: str = None,
        colab_train_df_path: str = None,
        colab_val_df_path: str = None
):
    # Load, merge and shuffle datasets
    isot_df = None
    if isot_true_path is not None and isot_fake_path is not None:
        logger.info("Raw ISOT paths detected. Triggering ISOT loading and shuffling process...")
        isot_df = prepare_raw_isot(isot_true_path, isot_fake_path)
    else:
        logger.info("ISOT paths not provided. Pipeline will proceed using only the primary dataset.")

    master_df = load_and_merge_datasets(welfake_path, isot_df)

    # Creating Stratified Splits (70/15/15)
    logger.info('Creating Stratified splits of master_df')
    train_df, val_df, test_df = create_stratified_splits(master_df)

    # Text Preprocessing
    logger.info('Starting text preprocessing pipeline')
    cleaner = TextCleaner(use_lemmatization=True)

    if colab_train_df_path is not None and colab_val_df_path is not None:
        logger.info("Loading pre-cleaned datasets exported from Colab...")
        train_df = pd.read_csv(colab_train_df_path)
        val_df = pd.read_csv(colab_val_df_path)
    else:
        train_df['cleaned_text'] = cleaner.pipeline(train_df['text'].tolist())
        val_df['cleaned_text'] = cleaner.pipeline(val_df['text'].tolist())

    train_df['cleaned_text'] = train_df['cleaned_text'].replace(r'^\s*$', pd.NA, regex=True)
    val_df['cleaned_text'] = val_df['cleaned_text'].replace(r'^\s*$', pd.NA, regex=True)

    train_df = train_df.dropna(subset=['cleaned_text']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['cleaned_text']).reset_index(drop=True)

    # feature Extraction (IF-IDF)
    logger.info('Executing IF-IDF features...')
    engineer = FeatureEngineer(max_features=50000)
    X_train = engineer.fit_transform(train_df['cleaned_text'].tolist())
    X_val = engineer.transform(val_df['cleaned_text'].tolist())

    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Ensure a directory exist To save the model
    os.makedirs('../artifacts', exist_ok=True)
    engineer.save_vectorizer('../artifacts/tfidf_vectorizer.joblib')

    # Train and evaluate model 1: Logistic Regression
    logger.info('Training logistic regression baseline model...')
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_val)

    # Check if predict_proba is available for AUC-ROC calculation
    lr_probs = lr_model.predict_proba(X_val)[:, 1]

    lr_metrics = compute_classification_matrix(y_val, lr_preds, y_prob=lr_probs)
    print(f'Logistic Regression Validation Metrics:\n{lr_metrics}\n')
    joblib.dump(lr_model, '../artifacts/lr_model.joblib')

    # Train and evaluate model 2: Linear Support Vector Classifier (LinerSVC)
    logger.info('Linear Support Vector Classifier baseline model...')
    svc_model = LinearSVC(random_state=42)
    svc_model.fit(X_train, y_train)
    svc_preds = svc_model.predict(X_val)
    svc_score = svc_model.decision_function(X_val)

    svc_metrics = compute_classification_matrix(y_val, svc_preds, y_prob=svc_score)
    print(f'Linear Support Vector Classification Metrics:\n{svc_metrics}\n')
    joblib.dump(svc_model, '../artifacts/svc_model.joblib')


    lr_f1 = lr_metrics.get("f1", 0) if lr_metrics is not None else 0
    svc_f1 = svc_metrics.get("f1", 0) if svc_metrics is not None else 0

    if lr_f1 > svc_f1:
        logger.info("Logistic Regression identified as the top baseline model.")
        top_preds = lr_preds
        top_model_name = "Logistic Regression"
    else:
        logger.info("LinearSVC identified as the top baseline model.")
        top_preds = svc_preds
        top_model_name = "LinearSVC"

    # Generate and save a confusion matrix plot for our top baseline model
    generate_confusion_matrix_plot(
        y_val,
        top_preds,
        f"../artifacts/top_baseline_confusion_matrix.png"
    )
    logger.info("Baseline training loop successfully completed.")

if __name__ == '__main__':
    run_baseline_pipeline(
        welfake_path="../data/raw/WELFake.csv",
        isot_true_path="../data/raw/ISOT_True.csv",
        isot_fake_path="../data/raw/ISOT_Fake.csv",
        colab_train_df_path="../data/processed/processed_train_df.csv",
        colab_val_df_path="../data/processed/processed_val_df.csv"
    )