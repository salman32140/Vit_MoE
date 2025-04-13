import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModel


class ViTransformer(nn.Module):
    def __init__(self, model_name):
        super(ViTransformer, self).__init__()
        self.vit = AutoModel.from_pretrained(model_name)
    
    def forward(self, x):
        return self.vit(x)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1, weight_decay=1e-4):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_decay = weight_decay
        
    def forward(self, x):
        x = torch.nn.functional.gelu(self.layer1(x))
        x = self.dropout(x)
        return torch.softmax(self.layer2(x), dim=1)


class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1, weight_decay=1e-4):
        super(Gating, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, num_experts)
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_decay = weight_decay
        
    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=1)


class MoE(nn.Module):
    def __init__(self, **kwargs):
        super(MoE, self).__init__()
        self.arr_experts = kwargs["arr_experts"]
        self.top_k = kwargs["top_k"]
        self.addNoise = False
        self.vit = ViTransformer(model_name="google/vit-base-patch16-224-in21k")
        self.layer_norm = nn.LayerNorm(normalized_shape=self.vit.vit.config.hidden_size)
        self.gate = Gating(input_dim=kwargs["input_dim"], num_experts=len(self.arr_experts))
        self.experts = nn.ModuleList(self.arr_experts)
    
    def forward(self, x):
        x = self.vit(x)
        pooler_out = x['pooler_output']
        pooler_out = self.layer_norm(pooler_out)
        
        if self.addNoise:
            noise_level = 0.0  # Adjust noise level if needed
            noise = noise_level * torch.randn_like(pooler_out)
            pooler_out = pooler_out + noise
        
        gate_out = self.gate(pooler_out) 
        
        topk_values, topk_indices = torch.topk(gate_out, k=self.top_k, dim=1)
        topk_values = topk_values / topk_values.sum(dim=1, keepdim=True)
        expert_outputs = torch.stack([self.experts[idx](pooler_out) for idx in range(len(self.arr_experts))], dim=1)
        topk_expert_outputs = torch.gather(expert_outputs, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        result = torch.sum(topk_values.unsqueeze(-1) * topk_expert_outputs, dim=1)
        
        return result, gate_out, expert_outputs
