import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from cross_modal_attention import CrossModalAttention

class HatefulMemesClassifierWithCrossAttention(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', vit_model_name='google/vit-base-patch16-224'):
        super(HatefulMemesClassifierWithCrossAttention, self).__init__()

        # BERT 
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_fc1 = nn.Linear(768, 128)
        self.bert_fc2 = nn.Linear(128, 128)
        self.bert_dropout = nn.Dropout(0.3)

        # ViT
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_fc1 = nn.Linear(768, 128)
        self.vit_fc2 = nn.Linear(128, 128)
        self.vit_dropout = nn.Dropout(0.3)

        # Cross-modality attention (BERT <--> ViT)
        self.text_to_image_attention = CrossModalAttention(embed_dim=128, num_heads=8)
        self.image_to_text_attention = CrossModalAttention(embed_dim=128, num_heads=8)

        # Classification head
        self.fusion_fc1 = nn.Linear(256, 64)
        self.fusion_fc2 = nn.Linear(64, 2)  

    def forward(self, input_ids, attention_mask, pixel_values):
        # BERT forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]  
        bert_x = self.bert_dropout(bert_cls)
        bert_x = torch.relu(self.bert_fc1(bert_x))
        bert_x = torch.relu(self.bert_fc2(bert_x))  

        # ViT forward pass
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_cls = vit_outputs.last_hidden_state[:, 0, :]  
        vit_x = self.vit_dropout(vit_cls)
        vit_x = torch.relu(self.vit_fc1(vit_x))
        vit_x = torch.relu(self.vit_fc2(vit_x))  

        # Text attending to image
        attn_text_to_image = self.text_to_image_attention(
            query=bert_x.unsqueeze(0),  # Query is text embeddings
            key=vit_x.unsqueeze(0),     # Key is image embeddings
            value=vit_x.unsqueeze(0)    # Value is image embeddings
        ).squeeze(0)

        # Image attending to text
        attn_image_to_text = self.image_to_text_attention(
            query=vit_x.unsqueeze(0),   # Query is image embeddings
            key=bert_x.unsqueeze(0),    # Key is text embeddings
            value=bert_x.unsqueeze(0)   # Value is text embeddings
        ).squeeze(0)

        # Combine the attended representations (e.g., by concatenation)
        combined = torch.cat((attn_text_to_image, attn_image_to_text), dim=1)  
        combined = torch.relu(self.fusion_fc1(combined))
        logits = self.fusion_fc2(combined)

        return logits