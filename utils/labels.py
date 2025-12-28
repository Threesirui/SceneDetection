import json
from typing import Dict, List, Tuple
from .config import LABEL2ID_JSON


import re
from typing import Dict, List, Tuple
import pandas as pd

def parse_labels(raw: str) -> List[str]:
    """
    把单元格里的标签字符串解析成 list[str]
    兼容：空格、中文逗号、顿号、分号、斜杠等分隔符
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    # 把各种分隔符统一成 "、"
    s = re.sub(r"[\s，、;/]+", "、", s).strip("、")
    if not s:
        return []
    return [p.strip() for p in s.split("、") if p.strip()]

def build_label_vocab(df: pd.DataFrame, label_col: str) -> Tuple[List[str], Dict[str, int]]:
    """
    从整份数据构建标签表（不做任何同义归一）
    """
    all_labs = set()
    for x in df[label_col].tolist():
        for l in parse_labels(x):
            all_labs.add(l)
    labels = sorted(all_labs)
    lab2id = {l: i for i, l in enumerate(labels)}
    return labels, lab2id

def labels_to_multi_hot(labels: List[str], lab2id: Dict[str, int], num_labels: int) -> List[int]:
    vec = [0] * num_labels
    for l in labels:
        if l in lab2id:
            vec[lab2id[l]] = 1
    return vec


def build_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], List[str],Dict[str, str]]:
    unique_labels = sorted(list(set(labels)))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {str(i): lab for lab, i in label2id.items()}
    return label2id, unique_labels,id2label


def save_label2id(label2id: Dict[str, int], path: str = LABEL2ID_JSON):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


def load_label2id(path: str = LABEL2ID_JSON) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_id2label(label2id: Dict[str, int]):
    return {int(v): k for k, v in label2id.items()}