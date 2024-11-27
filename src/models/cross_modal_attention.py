import torch
import torch.nn as nn
from transformers import BertModel, ViTModel, AutoModel

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        # Multihead attention for cross-modal interaction
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # Apply cross-attention (text -> image or image -> text)
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        return attn_output