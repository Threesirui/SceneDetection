import os
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from . import dataset  # noqa: F401 只是为了确保包正确
from utils.config import EXCEL_WEB, EXCEL_META, MERGED_CSV
from utils.labels import build_label_mappings, save_label2id
from sklearn.model_selection import train_test_split
from utils.labels import build_label_vocab, labels_to_multi_hot, parse_labels
import numpy as np
import json
from data.dataset import TextDataset,MultiLabelTextDataset
import csv
@dataclass
class DataConfig:
    excel_path: str
    text_col: str = "应用的具体场景"
    label_col_contains: str = "场景归类"   # 会在列名中查找包含该关键字的列
    test_size: float = 0.1
    val_size: float = 0.1   # 从剩余训练集中再切一份做验证
    seed: int = 42
    max_length: int = 128

    # 新增：保存目录（为空则不保存）
    save_dir: str = "artifacts/data"


def _normalize_column_name(col: str) -> str:
    return col.split("\n")[0].strip()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_artifacts(save_dir: str, meta: dict,
                    train_df: pd.DataFrame, train_y: np.ndarray, train_idx: np.ndarray,
                    val_df: pd.DataFrame, val_y: np.ndarray, val_idx: np.ndarray,
                    test_df: pd.DataFrame, test_y: np.ndarray, test_idx: np.ndarray,
                    text_col: str):
    _ensure_dir(save_dir)
    train_df.to_json(os.path.join(save_dir,'trian.json'),orient="records", force_ascii=False, lines=True)
    test_df.to_json(os.path.join(save_dir,'test.json'),orient="records", force_ascii=False, lines=True)
    val_df.to_json(os.path.join(save_dir,'val.json'),orient="records", force_ascii=False, lines=True)

    # 1) 保存标签映射（推理时必须用同一个顺序）
    labels = meta["labels"]
    lab2id = meta["lab2id"]
    id2lab = {str(v): k for k, v in lab2id.items()}  # json key 只能是 str
    with open(os.path.join(save_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "lab2id": lab2id,
                "id2lab": id2lab,
                "num_labels": meta["num_labels"],
                "label_col": meta["label_col"],
                "text_col": text_col,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2) 保存 multi-hot（npy）
    np.save(os.path.join(save_dir, "train_y.npy"), train_y)
    np.save(os.path.join(save_dir, "val_y.npy"), val_y)
    np.save(os.path.join(save_dir, "test_y.npy"), test_y)

    # 3) 保存划分索引（可复现）
    np.save(os.path.join(save_dir, "train_index.npy"), train_idx)
    np.save(os.path.join(save_dir, "val_index.npy"), val_idx)
    np.save(os.path.join(save_dir, "test_index.npy"), test_idx)

    # 4) 保存文本（可选但很实用）
    train_df[[text_col]].to_csv(os.path.join(save_dir, "train_text.csv"), index=False, encoding="utf-8-sig")
    val_df[[text_col]].to_csv(os.path.join(save_dir, "val_text.csv"), index=False, encoding="utf-8-sig")
    test_df[[text_col]].to_csv(os.path.join(save_dir, "test_text.csv"), index=False, encoding="utf-8-sig")



def load_short_descrption_data(
    excel_meta: str = EXCEL_META
) -> pd.DataFrame:
    print(f"Reading: {excel_meta}")
    df_meta = pd.read_excel(excel_meta)
    df_meta.columns = [_normalize_column_name(c) for c in df_meta.columns]
    if "场景归类" not in df_meta.columns:
        print("meta 列名：", df_meta.columns)
        raise ValueError("请确认 Excel 中 '场景归类' 列名。")
    if "应用的具体场景" not in df_meta.columns:
        raise ValueError(f"找不到 '来源' 列，当前列名：{df_meta.columns}")

    df = df_meta[["应用的具体场景", "场景归类"]]
    df = df.dropna(subset=["应用的具体场景", "场景归类"])
    df["应用的具体场景"] = df["应用的具体场景"].astype(str).str.strip()
    df["场景归类"] = df["场景归类"].astype(str).str.strip()
    df = df[df["应用的具体场景"] != ""]
    df = df[df["场景归类"] != ""]
    print("数据量：", len(df))
    print("场景类别分布：")
    print(df["场景归类"].value_counts())
    return df

 

def load_and_merge_data(
    excel_web: str = EXCEL_WEB,
    excel_meta: str = EXCEL_META
) -> pd.DataFrame:
    print(f"Reading: {excel_web}")
    df_web = pd.read_excel(excel_web)
    df_web.columns = [_normalize_column_name(c) for c in df_web.columns]

    print(f"Reading: {excel_meta}")
    df_meta = pd.read_excel(excel_meta)
    df_meta.columns = [_normalize_column_name(c) for c in df_meta.columns]
    
    if "来源" not in df_meta.columns:
        raise ValueError(f"找不到 '来源' 列，当前列名：{df_meta.columns}")
    if "网址" not in df_web.columns:
        raise ValueError(f"找不到 '网址' 列，当前列名：{df_web.columns}")
    if "网页内容" not in df_web.columns:
        raise ValueError(f"找不到 '网页内容' 列，当前列名：{df_web.columns}")
    if "场景归类" not in df_meta.columns:
        print("meta 列名：", df_meta.columns)
        raise ValueError("请确认 Excel 中 '场景归类' 列名。")
    #去重
    df_meta=df_meta.drop_duplicates(subset=['来源']).reset_index()
    #选一个标签
    print(len(df_meta))
    for i in range(len(df_meta)):
        label_content=df_meta.iloc[i]["场景归类"]
        label_content=label_content.split('、')[0]
        df_meta.loc[i,'场景归类']=label_content

    df_meta.dropna(how='any').reset_index()

    df = df_meta.merge(df_web, left_on="来源", right_on="网址", how="left")
    df = df[["网页内容", "场景归类"]]
    df = df.dropna(subset=["网页内容", "场景归类"])
    df["网页内容"] = df["网页内容"].astype(str).str.strip()
    df["场景归类"] = df["场景归类"].astype(str).str.strip()
    df = df[df["网页内容"] != ""]
    df = df[df["场景归类"] != ""]

    print("合并后的数据量：", len(df))
    print("场景类别分布：")
    print(df["场景归类"].value_counts())
    return df

def find_label_col(df: pd.DataFrame, contains: str) -> str:
    for c in df.columns:
        if contains in str(c):
            return c
    raise ValueError(f"找不到包含 '{contains}' 的标签列，当前列：{df.columns.tolist()}")


def preprocess_and_save(description : str = "short",cfg: DataConfig=None):
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    if description=='long':
        df = load_and_merge_data()
        label_col=find_label_col(df,"场景归类")
        text_col = "网页内容"
        label2id, labels,id2label = build_label_mappings(df["场景归类"].tolist())
        # save_label2id(label2id)
        df.to_csv(os.path.join(cfg.save_dir,'merged_data.csv'), index=False, encoding="utf-8-sig")
        print(f"预处理完成，数据已保存到 {MERGED_CSV}")

        num_labels = len(labels)
        meta = {
            "label_col": label_col,
            "labels": labels,
            "lab2id": label2id,
            "num_labels": num_labels,
        }

        with open(os.path.join(cfg.save_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": labels,
                    "lab2id": label2id,
                    "id2lab": id2label,
                    "num_labels": meta["num_labels"],
                    "label_col": meta["label_col"],
                    "text_col": text_col,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        
        return meta


    elif description=='short':
        df = load_short_descrption_data()
        label_col=find_label_col(df, "场景归类")
        labels,lab2id= build_label_vocab(df,label_col)

        num_labels = len(labels)
        #生成multi-hot
        y = []
        for raw in df[label_col].tolist():
            labs = parse_labels(raw)
            y.append(labels_to_multi_hot(labs, lab2id, num_labels))
        y = np.array(y, dtype=np.float32)

        # 划分：先 test，再 val
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True
        )

        def subset(idxs) -> Tuple[pd.DataFrame, np.ndarray]:
            return df.iloc[idxs].reset_index(drop=True), y[idxs]

        train_df, train_y = subset(train_idx)
        val_df, val_y = subset(val_idx)
        test_df, test_y = subset(test_idx)

        meta = {
            "label_col": label_col,
            "labels": labels,
            "lab2id": lab2id,
            "num_labels": num_labels,
        }

        # ✅ 保存 artifacts
        if cfg.save_dir:
            _save_artifacts(
                cfg.save_dir, meta,
                train_df, train_y, train_idx,
                val_df, val_y, val_idx,
                test_df, test_y, test_idx,
                text_col=cfg.text_col
            )

        return (train_df, train_y), (val_df, val_y), (test_df, test_y), meta

def dataset_jsonl_transfer(origin_path,is_multilabel):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    if is_multilabel:
        datatail=['train.json','test.json']
        for name in datatail:
            origin_paths=os.path.join(origin_path,name)
            path=name.split('.')[0]+'_sft_multi.json'
            new_path=os.path.join(origin_path,'sft',path)
            messages = []
            catagory=[]

            # 读取旧的JSONL文件
            with open(origin_paths, "r") as file:
                for line in file:
                    # 解析每一行的json数据
                    data = json.loads(line)
                    context = data["应用的具体场景"]
                    catagory = [ 
                        "事故处理",
                        "停车管控",
                        "安全引导",
                        "拥堵治理",
                        "流量检测",
                        "流量监测",
                        "车辆违法"
                    ]
                    label = data["场景归类"]
                    cat_str = "[" + ", ".join(catagory) + "]"
                    label=label.replace('、',',')
                    message = {
                        "instruction": "你是一个文本多标签分类专家。给定一段文本和一组备选标签，请从中选择所有适用的标签，并且只输出这些标签的列表，用逗号分隔；如果没有任何标签适用，则输出“无”,注意：只能从备选标签中选择，不要生成新的标签。",
                        "input": f"文本:{context},类型选型:{cat_str}",
                        "output": label,
                    }
                    messages.append(message)

            # 保存重构后的JSONL文件
            with open(new_path, "w", encoding="utf-8") as file:
                for message in messages:
                    file.write(json.dumps(message, ensure_ascii=False) + "\n")
    else:
        
        origin_paths=os.path.join(origin_path,'long_description','merged_data.csv')
        new_path=os.path.join(origin_path,'sft')
        messages = []
        catagory=[]
        
        with open(origin_paths, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)  # 读取CSV文件
            for row in reader:
                # 获取每一行的数据
                context = row["网页内容"]
                label = row["场景归类"]
                
                # 创建标签字符串
                cat_str = "[" + ", ".join(catagory) + "]"
                
                # 处理label中的“、”符号，替换为逗号
                label = label.replace('、', ',')
                
                # 创建消息
                message = {
                    "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。",
                    "input": f"文本:{context},类型选型:{cat_str}",
                    "output": label,
                }
                
                # 添加到消息列表
                messages.append(message)
        datas=int(len(messages)*0.8)
        messages_train=messages[0:datas]
        messages_test=messages[datas:-1]
        # 保存重构后的JSONL文件
        with open(os.path.join(new_path,'train_sft_single.json'), "w", encoding="utf-8") as file:
            for message in messages_train:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")
        # 保存重构后的JSONL文件
        with open(os.path.join(new_path,'test_sft_single.json'), "w", encoding="utf-8") as file:
            for message in messages_test:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")
        
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n 你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def process_func_multilabel(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n 你是一个文本多标签分类专家。给定一段文本和一组备选标签，请从中选择所有适用的标签，并且只输出这些标签的列表，用逗号分隔；如果没有任何标签适用，则输出“无”,注意：只能从备选标签中选择，不要生成新的标签。<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}  