import argparse
from modelscope import snapshot_download, AutoTokenizer
import os
from dataclasses import dataclass
from transformers import BertTokenizer
from utils.config import EXCEL_META
from data.preprocess import preprocess_and_save, dataset_jsonl_transfer,process_func
from data.loader_from_artifacts import ArtifactLoadConfig, build_dataloaders
from transformers import AutoTokenizer, get_linear_schedule_with_warmup,AutoModelForCausalLM,DataCollatorForSeq2Seq,Trainer,TrainingArguments
from models.bert_multilabel import ModelConfig, BertForMultiLabel
from models.bert_LSTM import BertLSTM
import torch
import json
from tqdm import tqdm
from utils.metrics import multilabel_metrics,classification_metrics
from datasets import Dataset
import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model

if torch.cuda.is_available():
    print("GPU可用，深度学习加速之旅开始！")
    device = torch.device("cuda:0")
    
else:
    print("GPU不可用，将使用CPU进行计算。")
    device = torch.device("cpu")


@dataclass
class DataConfig:
    excel_path: str=EXCEL_META
    text_col: str = "应用的具体场景"
    label_col_contains: str = "场景归类"   # 会在列名中查找包含该关键字的列
    test_size: float = 0.1
    val_size: float = 0.1   # 从剩余训练集中再切一份做验证
    seed: int = 42
    max_length: int = 128

    # 新增：保存目录（为空则不保存）
    save_dir: str = "artifacts/data"

def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

@torch.no_grad()
def evaluate(model, loader,is_multilabel, device):
    model.eval()
    all_probs, all_true = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        batch = move_to_device(batch, device)
        res = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
        )
        loss = res["loss"]
        logits = res["logits"]
        if is_multilabel:

            probs = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probs = torch.softmax(logits,dim=1).detach().cpu().numpy()
        y_true = batch["labels"].detach().cpu().numpy()

        total_loss += float(loss.item()) * len(y_true)
        n += len(y_true)
        all_probs.append(probs)
        all_true.append(y_true)

    all_probs = np.concatenate(all_probs, axis=0)
    all_pred = all_probs.argmax(axis=1)
    all_true = np.concatenate(all_true, axis=0)
    if is_multilabel:
        metrics = multilabel_metrics(all_true, all_probs, threshold=0.5)
    else:
        metrics =classification_metrics(all_true,all_pred)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="场景分类项目 (BERT / BertLSTM / GLM)")
    parser.add_argument(
        "--mode",
        type=str,
        default='train'
    )
    # parser.add_argument("--text", type=str, help="单条待预测文本")
    parser.add_argument("--description",type=str,default="short",help="标签对应的描述,短的描述或者网页描述")
    parser.add_argument("--model",type=str,default='BertLSTM', help="选择训练的模型BERT / BertLSTM/")
    parser.add_argument("--is_multilabel",type=bool,default=True,help="是否多标签分类")
    args = parser.parse_args()

  

    if args.mode == "process_data":
        if args.description=='long':
            cfg=DataConfig(
                excel_path =EXCEL_META,
                text_col = "网址内容",
                label_col_contains= "场景归类",  # 会在列名中查找包含该关键字的列
                test_size = 0.1,
                val_size= 0.1 ,  # 从剩余训练集中再切一份做验证
                seed = 42,
                max_length = 1024,
                # 新增：保存目录（为空则不保存）
                save_dir = "artifacts/data/long_description"
            )
            preprocess_and_save(args.description,cfg)
        elif args.description=="short":
            cfg=DataConfig(
                excel_path=EXCEL_META,
                text_col="应用的具体场景",
                label_col_contains = "场景归类"  , # 会在列名中查找包含该关键字的列
                test_size = 0.1,
                val_size = 0.1 ,  # 从剩余训练集中再切一份做验证
                seed =42,
                max_length = 128,
                # 新增：保存目录（为空则不保存）
                save_dir = "artifacts/data/short_description"
            )
            preprocess_and_save(args.description,cfg)
        return




    if args.mode=="train":
        
        if args.description == 'long':
            dl_cfg = ArtifactLoadConfig(
            artifacts_dir="artifacts/data/long_description",
            pretrained_name="bert-base-chinese",
            batch_size=16,
            max_length=512,
            num_workers=2,
            dataset_type= 'long'
            )
            tokenizer = AutoTokenizer.from_pretrained(dl_cfg.pretrained_name)
            train_loader, val_loader,test_loader, meta = build_dataloaders(dl_cfg)
             #读取模型和数据
            if args.model=="Bert":

                pretrained = "bert-base-chinese"
                batch_size = 16
                lr = 2e-5
                epochs = 100
                max_length = 128
                out_dir = "outputs/Single/Bert"
                os.makedirs(out_dir, exist_ok=True)


                # 初始化 BERT 多标签模型
                model_cfg = ModelConfig(
                    pretrained_name="bert-base-chinese",
                    num_labels=meta["num_labels"],
                    dropout=0.1,
                    freeze_bert=False,
                    loss_type="cross_entropy",
                )
                model = BertForMultiLabel(model_cfg).to(device)
            elif args.model=="BertLSTM":
                out_dir = "outputs/Single/BertLSTM"
                epochs = 100
                batch_size = 16
                lr = 2e-5
                os.makedirs(out_dir, exist_ok=True)
                train_loader, val_loader,test_loader, meta = build_dataloaders(dl_cfg)
                num_classes=meta['num_labels']

                model_cfg = ModelConfig(
                    pretrained_name="bert-base-chinese",
                    num_labels=meta["num_labels"],
                    # dropout=0.1,
                    freeze_bert=False,
                    loss_type="cross_entropy",
                )
                
                model = BertLSTM(
                    # bert_name="bert-base-chinese",
                    cfg=model_cfg,
                    hidden_dim=256,
                    num_classes=num_classes
                    # dropout=0.3
                )
                model.to(device)




            # AdamW
            optim = torch.optim.AdamW(model.parameters(), lr=lr)
            total_steps = epochs * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(
                optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )

            best_macro_f1 = -1.0
            best_path = os.path.join(out_dir, "best.pt")

            # 保存 label 信息
            with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": meta["labels"], "lab2id": meta["lab2id"]}, f, ensure_ascii=False, indent=2)

            for epoch in range(1, epochs + 1):
                model.train()
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
                total_loss, n = 0.0, 0

                for batch in pbar:
                    batch = move_to_device(batch, device)
                    res = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids"),
                        labels=batch["labels"],
                    )
                    loss = res["loss"]

                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    scheduler.step()

                    bs = batch["labels"].size(0)
                    total_loss += float(loss.item()) * bs
                    n += bs
                    pbar.set_postfix(loss=total_loss / max(n, 1))

                val_metrics = evaluate(model, val_loader,args.is_multilabel ,device)
                print(f"[VAL] {val_metrics}")

                # 按 macro_f1 选 best
                if val_metrics["f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["f1"]
                    torch.save({"model": model.state_dict(), "cfg": model_cfg.__dict__, "meta": meta}, best_path)
                    print(f"Saved best to {best_path} (macro_f1={best_macro_f1:.4f})")

            # test
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            test_metrics = evaluate(model, test_loader, args.is_multilabel, device)
            print(f"[TEST] {args.model}-{args.description}:{test_metrics}")
            # elif args.model=="Bert_LSTM":
        
        if args.description == 'short':
        
            pretrained="bert-base-chinese"
            lr = 2e-5
            epochs = 100
            dl_cfg = ArtifactLoadConfig(
                artifacts_dir="artifacts/data/short_description",
                pretrained_name="bert-base-chinese",
                batch_size=16,
                max_length=128,
                num_workers=2,
                dataset_type='short'
                )
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
            train_loader, val_loader, test_loader, meta = build_dataloaders(dl_cfg)

            #读取模型和数据
            if args.model=="Bert":

                pretrained = "bert-base-chinese"
                batch_size = 16
                lr = 2e-5
                epochs = 100
                max_length = 128
                out_dir = "outputs/Multi/Bert"
                os.makedirs(out_dir, exist_ok=True)

               

                # 初始化 BERT 多标签模型
                model_cfg = ModelConfig(
                    pretrained_name="bert-base-chinese",
                    num_labels=meta["num_labels"],
                    dropout=0.1,
                    freeze_bert=False,
                    loss_type="bce",
                    
                )
                model = BertForMultiLabel(model_cfg).to(device)
            elif args.model=="BertLSTM":
                out_dir = "outputs/Multi/BertLSTM"
                os.makedirs(out_dir, exist_ok=True)
                train_loader, val_loader, test_loader, meta = build_dataloaders(dl_cfg)
                num_classes=meta['num_labels']

                model_cfg = ModelConfig(
                    pretrained_name="bert-base-chinese",
                    num_labels=meta["num_labels"],
                    # dropout=0.1,
                    freeze_bert=False,
                    loss_type="bce",
                )
                
                model = BertLSTM(
                    # bert_name="bert-base-chinese",
                    cfg=model_cfg,
                    hidden_dim=256,
                    num_classes=num_classes
                    # dropout=0.3
                )
                model.to(device)




            # AdamW
            optim = torch.optim.AdamW(model.parameters(), lr=lr)
            total_steps = epochs * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(
                optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )

            best_macro_f1 = -1.0
            best_path = os.path.join(out_dir, "best.pt")

            # 保存 label 信息
            with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": meta["labels"], "lab2id": meta["lab2id"]}, f, ensure_ascii=False, indent=2)

            for epoch in range(1, epochs + 1):
                model.train()
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
                total_loss, n = 0.0, 0

                for batch in pbar:
                    batch = move_to_device(batch, device)
                    res = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        token_type_ids=batch.get("token_type_ids"),
                        labels=batch["labels"],
                    )
                    loss = res["loss"]

                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    scheduler.step()

                    bs = batch["labels"].size(0)
                    total_loss += float(loss.item()) * bs
                    n += bs
                    pbar.set_postfix(loss=total_loss / max(n, 1))

                val_metrics = evaluate(model, val_loader,args.is_multilabel, device)
                print(f"[VAL] {val_metrics}")

                # 按 macro_f1 选 best
                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    torch.save({"model": model.state_dict(), "cfg": model_cfg.__dict__, "meta": meta}, best_path)
                    print(f"Saved best to {best_path} (macro_f1={best_macro_f1:.4f})")

            # test
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            test_metrics = evaluate(model, test_loader, args.is_multilabel,  device)
            print(f"[TEST] {args.model}-{args.description}:{test_metrics}")
            # elif args.model=="Bert_LSTM":


        



if __name__ == "__main__":
    main()
