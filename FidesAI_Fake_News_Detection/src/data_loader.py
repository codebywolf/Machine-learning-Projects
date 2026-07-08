import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_raw_isot(isot_true_path: str, isot_fake_path: str):
    isot_true_df = pd.read_csv(isot_true_path)
    isot_true_df['label'] = 1
    isot_fake_df = pd.read_csv(isot_fake_path)
    isot_fake_df['label'] = 0

    df_isot_combined = pd.concat([isot_true_df, isot_fake_df], ignore_index=True)
    df_isot_combined = df_isot_combined[['text', 'label']]
    # Shuffling 5 times using a loop with varying random seeds
    for i in range(5):
        # Generates a new seed for each pass (e.g., 42, 43, 44, 45, 46)
        current_seed = 42 + i
        df_isot_combined = df_isot_combined.sample(frac=1, random_state=current_seed).reset_index(drop=True)
    df_isot_combined.dropna(inplace=True)
    df_isot_combined.drop_duplicates(inplace=True)
    df_isot_combined = df_isot_combined.reset_index(drop=True)
    return df_isot_combined


def load_and_merge_datasets(wakfake_pth:str, isot_dataset: pd.DataFrame = None):
    logger.info(f'Loading WAKFake dataset...')
    wf_df = pd.read_csv(wakfake_pth)
    wf_df = wf_df[['text', 'label']].dropna().reset_index(drop=True)

    # WELFake convention: 1 = Fake, 0 = Real
    # Remapping to unified convention: 1=Real, 0=Fake (matches ISOT)
    wf_df['label'] = wf_df['label'].map({0: 1, 1: 0})

    if isot_dataset is not None:
        logger.info(f'Combining WAKFake and ISOT datasets into one dataset...')
        df_combined = pd.concat([wf_df, isot_dataset], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)
        df_combined = df_combined.reset_index(drop=True)
        return df_combined

    return wf_df

def create_stratified_splits(
        df: pd.DataFrame,
        target_col: str = 'label',
        test_size: float = 0.15,
        val_size: float = 0.15
):
    # Creating a strict 70:15:15 stratified train/val/test split to preserve class distribution.
    logger.info("Executing stratified train/test/val splits...")

    combined_test_size = test_size + val_size
    train_df, temp_df = train_test_split(
        df,
        test_size=combined_test_size,
        stratify=df[target_col],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[target_col],
        random_state=42
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split completed. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


class FakeNewsDataset(Dataset):
    # Unified PyTorch Dataset to tokenize text sequences on-the-fly
    # and package them into numeric tensors for RoBERTa.

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }