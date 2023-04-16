from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    max_input_length: int
    num_heads: int = 8
    num_blocks: int = 6
    embedding_size: int = 512
    masked: bool = True
    p_dropout: float = 0.1

class Embedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=self.config.vocab_size, embedding_dim=self.config.embedding_size)
        # Register as parameter to move to gpu with "to" method
        self.encoding_vector = nn.Parameter(self.compute_encoding(), requires_grad=False)
        self.drop = nn.Dropout(p=config.p_dropout)

    def compute_encoding(self):
        encoding_vector = torch.zeros((self.config.max_input_length, self.config.embedding_size))
        for i in range(self.config.embedding_size // 2):
            encoding_vector[:, 2*i] = torch.sin(torch.arange(0, self.config.max_input_length)/(10000**(2*i/self.config.embedding_size)))
            encoding_vector[:, 2*i+1] = torch.cos(torch.arange(0, self.config.max_input_length)/(10000**(2*i/self.config.embedding_size)))
        return encoding_vector

    def forward(self, x):
        # Input dimensions [batch_size, sequence_length]
        num_tokens = x.shape[1]
        x = self.embedding(x)
        assert x.shape[1:] == self.encoding_vector[:num_tokens].shape
        # Encoding will be broadcast across batch dimension
        x = x + self.encoding_vector[:num_tokens]
        return self.drop(x)

class Attention(nn.Module):
    def __init__(self, output_size, config: TransformerConfig):
        super().__init__()
        self.config = config
        if output_size == None:
            self.output_size = self.config.embedding_size
        else:
            self.output_size = output_size
        # What initialization?
        self.WQ = nn.Parameter(torch.empty(self.config.embedding_size, self.output_size))
        self.WK = nn.Parameter(torch.empty(self.config.embedding_size, self.output_size))
        self.WV = nn.Parameter(torch.empty(self.config.embedding_size, self.output_size))
        nn.init.xavier_normal_(self.WQ)
        nn.init.xavier_normal_(self.WK)
        nn.init.xavier_normal_(self.WV)
            
    def forward(self, x):
        # Dimension [batch, sequence_length, embedding_dim]
        # Matmul will be broadcast along batch dimension
        Q = torch.matmul(x, self.WQ)
        K = torch.matmul(x, self.WK)
        V = torch.matmul(x, self.WV)
        # BMM: batched matrix multiplication
        scaled_score = torch.bmm(Q, torch.transpose(K, 1, 2))/np.sqrt(self.config.embedding_size)
        if self.config.masked:
            mask = torch.ones((x.shape[1], x.shape[1])).triu(diagonal=1)
            scaled_score[:, mask>0] = float("-inf")
        normalized_score = F.softmax(scaled_score, dim=2)
        return torch.bmm(normalized_score, V)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(config.embedding_size, config.embedding_size))
        nn.init.xavier_normal_(self.W0)
        output_size = config.embedding_size // config.num_heads
        assert config.embedding_size % config.num_heads == 0
        self.heads =  nn.ModuleList([Attention(output_size, config) for _ in range(config.num_heads)])

    def forward(self, x):
        # Sequential forward pass
        head_results = [head(x) for head in self.heads]
        x = torch.cat(head_results, dim=2)
        x = torch.matmul(x, self.W0)
        return x
    
class PositionalFF(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.f1 = nn.Linear(config.embedding_size, config.embedding_size*4)
        self.f2 = nn.Linear(config.embedding_size*4, config.embedding_size)
    
    def forward(self, x):
        # Broadcast will automatically apply the ff positional-wise
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_block = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.embedding_size)
        self.ff_block = PositionalFF(config)
        self.norm2 = nn.LayerNorm(config.embedding_size)
        self.drop1 = nn.Dropout(p=config.p_dropout)
        self.drop2 = nn.Dropout(p=config.p_dropout)
    
    def forward(self, x):
        x = self.norm1(self.drop1(self.attention_block(x))+x)
        x = self.norm2(self.drop2(self.ff_block(x))+x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, apply_softmax=True):
        super().__init__()
        self.embedding = Embedding(config)
        self.apply_softmax = apply_softmax
        self.blocks =  nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
    
    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.blocks:
            x = transformer_block(x)
        # Reuse embedding weights
        x = torch.matmul(x, self.embedding.embedding.weight.T)
        if self.apply_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x