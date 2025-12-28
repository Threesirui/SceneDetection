import argparse
from modelscope import snapshot_download, AutoTokenizer
import os
from transformers import BertTokenizer
from data.preprocess import dataset_jsonl_transfer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup,AutoModelForCausalLM,DataCollatorForSeq2Seq,Trainer,TrainingArguments
import torch
import json
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

if torch.cuda.is_available():
    print("GPU可用，深度学习加速之旅开始！")
    device = torch.device("cuda:0")
    
else:
    print("GPU不可用，将使用CPU进行计算。")
    device = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="无人机场景分类项目 (BERT / LSTM / Claude)")
    parser.add_argument(
        "--mode",
        type=str,
        default='train'
    )
    parser.add_argument("--text", type=str, help="单条待预测文本")
    parser.add_argument("--description",type=str,default="long",help="标签对应的描述")
    parser.add_argument("--model",type=str,default='zhipuai', help="选择训练的模型")
    parser.add_argument("--is_multilabel",type=bool,default=False,help="是否多标签分类")
    parser.add_argument("--out_dir",type=str,default="./output/GLM4-9b-single",help="多标签")
    parser.add_argument("--checkpoint",type=str,default="./output/GLM4-9b-single/checkpoint-460")
    args = parser.parse_args()

    artifacts_dir = "artifacts/data"
    origin_path='./artifacts/data/'
    if args.mode=='train':
        
        model_dir = snapshot_download("ZhipuAI/glm-4-9b-chat", cache_dir="./model_download", revision="master")
        tokenizer = AutoTokenizer.from_pretrained("./moedel_download/ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("./moedel_download/ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        # api_key = os.getenv("ZHIPUAI_API_KEY")
        
            
        if args.is_multilabel:
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
        


            dataset_jsonl_transfer(origin_path,args.is_multilabel)
            train_df = pd.read_json(os.path.join(origin_path,'sft','train_sft_multi.json'), lines=True)
            train_ds= Dataset.from_pandas(train_df)
            # split=ds.train_test_split(test_size=0.2)
            # train_dataset,test_dataset=split['train'],split['test']
            train_dataset = train_ds.map(process_func_multilabel, remove_columns=train_ds.column_names)
        else :
            def process_func(example):
                """
                将数据集进行预处理
                """
                MAX_LENGTH = 1024
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
            dataset_jsonl_transfer(origin_path,args.is_multilabel)
            df = pd.read_json(os.path.join(origin_path,'sft','singlesft.json'), lines=True)

            ds = Dataset.from_pandas(df)
            split=ds.train_test_split(test_size=0.2)
            train_ds,test_ds=split['train'],split['test']

            # train_ds = Dataset.from_pandas(df)
            train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1,  # Dropout 比例
        )

        model = get_peft_model(model, config).to(device)

        train_args = TrainingArguments(
            output_dir=args.out_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=10,
            num_train_epochs=20,
            save_steps=100,
            learning_rate=1e-4,
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        trainer.train() 

        def predict(messages, model, tokenizer):
            device = "cuda"
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(response)
            
            return response
    if args.mode=='test':
        tokenizer = AutoTokenizer.from_pretrained("./moedel_download/ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("./moedel_download/ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(model, model_id=args.checkpoint)
        def predict(messages, model, tokenizer):
            device = "cuda"
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(response)
            
            return response
        if args.is_multilabel:
            test_jsonl_new_path=os.path.join(origin_path,'sft','test_sft_multi.json')
        else:
            test_jsonl_new_path=os.path.join(origin_path,'sft','test_sft_single.json')
        test_df = pd.read_json(test_jsonl_new_path, lines=True)

        label_list = []
        pred_list=[]
        for index, row in test_df.iterrows():
            instruction = row['instruction']
            input_value = row['input']
            label=row['output']
            label_list.append(label)
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]

            response = predict(messages, model, tokenizer)
            response=response.replace("\n",'')
            pred_list.append(response)
            messages.append({"role": "assistant", "content": f"{response}"})
            result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
            # test_text_list.append(result_text)
            # test_text_list.append(swanlab.Text(result_text, caption=response))
        if args.is_multilabel:
            all_labels=["事故处理",
                "停车管控",
                "安全引导",
                "拥堵治理",
                "流量检测",
                "流量监测",
                "车辆违法"
            ]
            def split_labels(s: str):
                if not s or s.strip() == "无":
                    return []
                return [x.strip() for x in s.replace("、", ",").split(",") if x.strip()]

            y_true = [split_labels(x) for x in label_list]
            y_pred = [split_labels(x) for x in pred_list]
            mlb = MultiLabelBinarizer(classes=all_labels)
            Y_true = mlb.fit_transform(y_true)
            Y_pred = mlb.transform(y_pred)
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                Y_true, Y_pred, average="micro", zero_division=0
            )

            # macro（对小类更公平）
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                Y_true, Y_pred, average="macro", zero_division=0
            )
            subset_accuracy = accuracy_score(Y_true, Y_pred)
            print(f"Micro  P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
            print(f"Macro  P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
            print(f"Subset Accuracy: {subset_accuracy:.4f}")
            rows = []
            for i, lab in enumerate(all_labels):
                p, r, f1, _ = precision_recall_fscore_support(
                    Y_true[:, i], Y_pred[:, i], average="binary", zero_division=0
                )
                support = Y_true[:, i].sum()
                rows.append((lab, p, r, f1, int(support)))

            df = pd.DataFrame(rows, columns=["label", "precision", "recall", "f1", "support"])
            print(df.sort_values("support", ascending=False))
        else:
            assert len(label_list) == len(pred_list)

            # 1) Accuracy
            acc = accuracy_score(label_list, pred_list)

            # 2) Precision / Recall / F1
            # macro: 各类平均（对小类更公平）
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                label_list, pred_list, average="macro", zero_division=0
            )

            # micro: 全局统计（更看重大类）
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                label_list, pred_list, average="micro", zero_division=0
            )

            # weighted: 按样本数加权（介于 macro 和 micro 之间）
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                label_list, pred_list, average="weighted", zero_division=0
            )

            print(f"Accuracy: {acc:.4f}")
            print(f"Macro   - P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
            print(f"Micro   - P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
            print(f"Weighted- P/R/F1: {p_w:.4f} / {r_w:.4f} / {f1_w:.4f}")
            print(classification_report(label_list, pred_list, zero_division=0))

if __name__ == "__main__":
    main()
