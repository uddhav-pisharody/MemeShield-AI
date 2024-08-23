import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel

class HatefulMemesClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', vit_model_name='google/vit-base-patch16-224'):
        super(HatefulMemesClassifier, self).__init__()

        # BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_fc1 = nn.Linear(768, 128)
        self.bert_fc2 = nn.Linear(128, 64)
        self.bert_dropout = nn.Dropout(0.3)

        # Vision Transformer (ViT) model
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
        self.vit_fc = nn.Linear(256, 64)  # 128*2 because of bidirectional LSTM
        self.vit_dropout = nn.Dropout(0.3)

        # Fusion and classification head
        self.fusion_fc1 = nn.Linear(128, 64)
        self.fusion_fc2 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask, pixel_values):
        # BERT forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        bert_x = self.bert_dropout(bert_cls)
        bert_x = torch.relu(self.bert_fc1(bert_x))
        bert_x = torch.relu(self.bert_fc2(bert_x))

        # ViT forward pass
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_cls = vit_outputs.last_hidden_state  # All tokens
        vit_x, _ = self.vit_lstm(vit_cls)
        vit_x = vit_x[:, -1, :]  # Take the last hidden state
        vit_x = self.vit_dropout(vit_x)
        vit_x = torch.relu(self.vit_fc(vit_x))

        # Fusion
        combined = torch.cat((bert_x, vit_x), dim=1)
        combined = torch.relu(self.fusion_fc1(combined))
        logits = self.fusion_fc2(combined)

        return logits