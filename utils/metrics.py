from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "subset_acc": float((y_true == y_pred).all(axis=1).mean()),
    }
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    单标签分类评价指标（适用于二分类或多分类）
    
    参数:
        y_true: numpy array, shape (N,) 真实标签（类别 id）
        y_pred: numpy array, shape (N,) 预测标签（类别 id）
    
    返回:
        dict : {accuracy, precision, recall, f1}
    """
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }