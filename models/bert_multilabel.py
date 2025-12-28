from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel

@dataclass
class ModelConfig:
    pretrained_name: str = "bert-base-chinese"
    num_labels: int = 6
    dropout: float = 0.1
    freeze_bert: bool = False
    loss_type : str = 'BCE'


class BertForMultiLabel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.pretrained_name)
        self.loss_type=cfg.loss_type
        if cfg.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden, cfg.num_labels)
        self.num_labels=cfg.num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 用 pooler_output（等价于 BERT 的 [CLS] 经过 tanh 的那层）
        pooled = out.pooler_output
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if self.loss_type=='bce':
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        elif self.loss_type=='cross_entropy':
            # 交叉熵损失（用于单标签分类）
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
