import torch
import torch.nn as nn
from transformers import BertModel

class BertLSTM(nn.Module):
    def __init__(self, cfg,hidden_dim=256, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.num_labels=num_classes
        self.loss_type=cfg.loss_type
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask,token_type_ids,labels):
        # BERT 输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # (B, L, 768)

        # LSTM
        lstm_out, _ = self.lstm(sequence_output)     # (B, L, hidden*2)

        # 取最后时刻的 hidden（或用 mean/max pooling）
        features = lstm_out[:, -1, :]                # (B, hidden*2)

        logits = self.classifier(features)
        loss = None
        if self.loss_type=='bce':
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        elif self.loss_type=='cross_entropy':
            # 交叉熵损失（用于单标签分类）
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}
