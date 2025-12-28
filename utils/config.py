import os
import torch

# ========= 路径配置 =========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始 Excel
EXCEL_WEB = os.path.join(PROJECT_ROOT, "网址＋纯文本网页内容.xlsx")
EXCEL_META = os.path.join(PROJECT_ROOT, "原始数据勿改！！！！_无人机在交管领域的应用调研.xlsx")

# 预处理输出
MERGED_CSV = os.path.join(PROJECT_ROOT, "artifacts", "long_description", "merged_data.csv")
LABEL2ID_JSON = os.path.join(PROJECT_ROOT, "artifacts", "long_description","label2id.json")

# 模型保存
BERT_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "bert_scene_cls.pt")
LSTM_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "lstm_scene_cls.pt")

# ========= 模型与训练配置 =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL_NAME = "bert-base-chinese"

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3

LR_BERT = 2e-5
LR_LSTM = 1e-3
