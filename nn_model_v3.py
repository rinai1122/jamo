import torch
import torch.nn as nn
import math

class FrequencyEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(FrequencyEncoder, self).__init__()
        # Encodes the unigram frequencies of a ciphertext into a vector
        self.fc = nn.Linear(vocab_size, d_model)
        
    def forward(self, src_indices, vocab_size):
        # src_indices: (S, B)
        batch_size = src_indices.size(1)
        freqs = torch.zeros(batch_size, vocab_size).to(src_indices.device)
        for b in range(batch_size):
            counts = torch.bincount(src_indices[:, b], minlength=vocab_size).float()
            freqs[b] = counts / len(src_indices[:, b])
        return self.fc(freqs) # (B, d_model)

class MetaTransformerCracker(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super(MetaTransformerCracker, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(100, 1, d_model)) # Fixed max len for simplicity
        
        # Frequency conditioning
        self.freq_encoder = FrequencyEncoder(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # src: (S, B)
        S, B = src.shape
        
        # 1. Frequency context
        key_context = self.freq_encoder(src, self.vocab_size) # (B, d_model)
        key_context = key_context.unsqueeze(0) # (1, B, d_model)
        
        # 2. Main embedding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoder[:S, :, :]
        
        # 3. Prepend context to sequence (like a class token or prefix)
        full_src = torch.cat([key_context, src_emb], dim=0) # (S+1, B, d_model)
        
        # 4. Transform
        output = self.transformer_encoder(full_src)
        
        # 5. Take the sequence parts (skip the context token)
        seq_output = output[1:] # (S, B, d_model)
        
        return self.out(seq_output)
