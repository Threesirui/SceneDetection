from dataclasses import dataclass
from typing import Dict, Tuple
from torch.utils.data import random_split
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data.dataset import MultiLabelTextDataset,TextDataset  # 复用你已有的 Dataset
import os

@dataclass
class ArtifactLoadConfig:
    artifacts_dir: str = "artifacts/data"
    pretrained_name: str = "bert-base-chinese"
    batch_size: int = 16
    max_length: int = 128
    num_workers: int = 2
    shuffle_train: bool = True
    dataset_type : str = 'long'


def load_artifacts(artifacts_dir: str) -> Dict:
    """
    读取 labels.json（标签顺序/映射 & 列名信息）
    """
    import os, json
    path = os.path.join(artifacts_dir, "labels.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def load_split_xy(artifacts_dir: str, split: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    split: train / val / test
    """
    import os

    text_path = os.path.join(artifacts_dir, f"{split}_text.csv")
    y_path = os.path.join(artifacts_dir, f"{split}_y.npy")

    df = pd.read_csv(text_path, encoding="utf-8-sig")
    y = np.load(y_path).astype(np.float32)

    if len(df) != len(y):
        raise ValueError(f"{split}: 文本条数({len(df)}) 与 y 条数({len(y)}) 不一致")
    return df, y


def build_dataloaders(cfg: ArtifactLoadConfig):
    if cfg.dataset_type =='long':
        meta =load_artifacts(cfg.artifacts_dir)
        # text_col = meta.get("text_col", "应用的具体场景")
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)
        df=pd.read_csv(os.path.join(cfg.artifacts_dir,'merged_data.csv'))
        datasets=TextDataset(df,meta['lab2id'],meta['text_col'],meta['label_col'],tokenizer=tokenizer,max_length=512)
        train_size=int(0.8*len(datasets))
        test_size=int(0.1*len(datasets))
        val_size=len(datasets)-train_size-test_size
        train_dataset,test_dataset,val_dataset=random_split(datasets,[train_size,test_size,val_size])

        # datasets=datasets.train_test_split(test_size=0.2,seed=42)
        # train_dataset,test_dataset=datasets['train'],datasets['test']
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        return train_loader,val_loader,test_loader,meta
        # tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)

        # train_size=int(0.8*len(datasets))
        # test_size=len(datasets)-train_size
        # train_dataset,test_dataset=random_split(datasets,[train_size,test_size])
    
    elif cfg.dataset_type =='short':
        """
        返回：train_loader, val_loader, test_loader, meta
        meta 里包含 labels/lab2id/num_labels/text_col 等
        """
        meta = load_artifacts(cfg.artifacts_dir)
        text_col=meta['text_col']

        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)

        train_df, train_y = load_split_xy(cfg.artifacts_dir, "train")
        val_df, val_y = load_split_xy(cfg.artifacts_dir, "val")
        test_df, test_y = load_split_xy(cfg.artifacts_dir, "test")

        train_ds = MultiLabelTextDataset(train_df, train_y, tokenizer, text_col, max_length=cfg.max_length)
        val_ds = MultiLabelTextDataset(val_df, val_y, tokenizer, text_col, max_length=cfg.max_length)
        test_ds = MultiLabelTextDataset(test_df, test_y, tokenizer, text_col, max_length=cfg.max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        return train_loader, val_loader, test_loader, meta
