from typing import List, Dict, Tuple
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
from utils.config import MERGED_CSV, MAX_LEN, BATCH_SIZE
from utils.labels import load_label2id
import numpy as np

class SceneDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class TextDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        lab2id: dict,
        text_col:str,
        label_col:str,
        tokenizer:AutoTokenizer,
        max_length: int = 128,
    ):
        self.df=df.reset_index()
        # self.lab2id= {l:i for i,l in enumerate(sorted(self.df["场景归类"].unique()))}
        self.lab2id=lab2id
        self.text_col=text_col
        self.text=df[text_col].to_list()
        self.label_col=label_col
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.text)
    def __getitem__(self,idx):
        text=self.df.loc[idx,self.text_col]
        # label=self.lab2id[self.df.loc[idx,label_col]]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.lab2id[self.df.loc[idx,self.label_col]]
        return item

class MultiLabelTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        tokenizer: AutoTokenizer,
        text_col: str,
        max_length: int = 128,
    ):
        self.texts = df[text_col].tolist()
        self.y = torch.tensor(y, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int):
        text = self.texts[i]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.y[i]
        return item

def load_data_and_label2id() -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(MERGED_CSV)
    label2id = load_label2id()
    return df, label2id


def create_dataloaders(
    df: pd.DataFrame,
    label2id: Dict[str, int],
    tokenizer: BertTokenizer,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE
):
    texts = df["网页内容"].tolist()
    labels = [label2id[lab] for lab in df["场景归类"].tolist()]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    train_ds = SceneDataset(X_train, y_train, tokenizer, max_len)
    val_ds = SceneDataset(X_val, y_val, tokenizer, max_len)
    test_ds = SceneDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

