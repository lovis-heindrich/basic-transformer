from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    max_input_length: int
    embedding_size: int = 512


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, max_input_length=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_input_length = max_input_length
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.encoding_vector = self.compute_encoding()

    def compute_encoding(self):
        encoding_vector = torch.zeros((self.max_input_length, self.embedding_size))
        for i in range(self.embedding_size // 2):
            encoding_vector[:, 2*i] = torch.sin(torch.arange(0, self.max_input_length)/(10000**(2*i/self.embedding_size)))
            encoding_vector[:, 2*i+1] = torch.cos(torch.arange(0, self.max_input_length)/(10000**(2*i/self.embedding_size)))
        return encoding_vector

    def forward(self, x):
        # Input dimensions [batch_size, sequence_length]
        num_tokens = x.shape[1]
        x = self.embedding(x)
        assert x.shape[1:] == self.encoding_vector[:num_tokens].shape
        # Encoding will be broadcast across batch dimension
        x = x + self.encoding_vector[:num_tokens]
        return x

class Attention(nn.Module):
    def __init__(self, embedding_size=512, output_size=None, masked=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.masked = masked
        if output_size == None:
            self.output_size = self.embedding_size
        else:
            self.output_size = output_size
        # What initialization?
        self.WQ = nn.Parameter(torch.empty(self.embedding_size, self.output_size))
        self.WK = nn.Parameter(torch.empty(self.embedding_size, self.output_size))
        self.WV = nn.Parameter(torch.empty(self.embedding_size, self.output_size))
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
        scaled_score = torch.bmm(Q, torch.transpose(K, 1, 2))/np.sqrt(self.embedding_size)
        if self.masked:
            mask = torch.ones((x.shape[1], x.shape[1])).triu(diagonal=1)
            scaled_score[:, mask>0] = float("-inf")
        normalized_score = F.softmax(scaled_score, dim=2)
        return torch.bmm(normalized_score, V)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size=512, num_heads=8, masked=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.W0 = nn.Parameter(torch.empty(embedding_size, embedding_size))
        nn.init.xavier_normal_(self.W0)
        output_size = embedding_size // num_heads
        assert embedding_size % num_heads == 0
        self.heads = [Attention(embedding_size, output_size, masked=masked) for _ in range(num_heads)]

    def forward(self, x):
        # Sequential forward pass
        head_results = [head(x) for head in self.heads]
        x = torch.cat(head_results, dim=2)
        x = torch.matmul(x, self.W0)
        return x
    
class PositionalFF(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.f1 = nn.Linear(embedding_size, embedding_size*4)
        self.f2 = nn.Linear(embedding_size*4, embedding_size)
    
    def forward(self, x):
        # Broadcast will automatically apply the ff positional-wise
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.attention_block = MultiHeadAttention(embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ff_block = PositionalFF(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        x = self.norm1(self.attention_block(x)+x)
        x = self.norm2(self.ff_block(x)+x)
        return x

class Transformer(nn.Module):
    def __init__(self, embedding_size=512, num_blocks=6, vocab_size=10, max_input_length=10):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_size, max_input_length)
        self.blocks = [TransformerBlock(embedding_size) for _ in range(num_blocks)]
    
    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.blocks:
            x = transformer_block(x)
        # Reuse embedding weights
        x = torch.matmul(x, self.embedding.embedding.weight.T)
        return F.softmax(x, dim=-1)