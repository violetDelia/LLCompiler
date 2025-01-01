import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
import torch._dynamo
from llcompiler.test_models import *
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.1,
):
    k_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)
    attention_score = F.softmax(scores, dim=-1)
    out = torch.matmul(attention_score, value)
    return scores,out
    return out, attention_score  # shape: (seq_len, v_dim), (seq_len, seq_lem)

class MultiHeadedAttention(nn.Module):
    def __init__(self, 
                num_heads: int= 2,
                d_model: int =8, 
                dropout: float=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.k_dim = d_model // num_heads
        self.num_heads = num_heads
        self.proj_weights = clones(nn.Linear(d_model, d_model), 4)
        self.attention_score = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, 
                query:Tensor, 
                key: Tensor, 
                value: Tensor, 
                mask:Optional[Tensor]=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        
        query, key, value \
            = [proj_weight(x).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
                for proj_weight, x in zip(self.proj_weights, [query, key, value])] 
        out, self.attention_score = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # out = out.transpose(1, 2).contiguous() \
        #      .view(batch_size, -1, self.num_heads * self.k_dim)
        # out = self.proj_weights[-1](out)
        return out

