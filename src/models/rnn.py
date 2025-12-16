"""
LSTM language model for comparison with transformers.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTMConfig:
    """Configuration for LSTM model."""
    
    def __init__(self, vocab_size, n_embd=128, n_layer=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.dropout = dropout


class LSTMModel(nn.Module):
    """LSTM-based language model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.lstm = nn.LSTM(
            config.n_embd,
            config.n_embd,
            config.n_layer,
            dropout=config.dropout if config.n_layer > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"LSTM parameters: {self.get_num_params():,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets=None, hidden=None):
        # Embedding
        x = self.embedding(idx)
        
        # LSTM
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        
        # Output projection
        logits = self.fc(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens."""
        hidden = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, hidden = self(idx if hidden is None else idx[:, -1:], hidden=hidden)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_lstm_model(model_size, vocab_size):
    """
    Factory function to create LSTM models of different sizes.
    
    Sizes chosen to roughly match transformer parameter counts.
    """
    
    configs = {
        'tiny': LSTMConfig(
            vocab_size=vocab_size,
            n_embd=256,
            n_layer=2,
            dropout=0.1
        ),
        'small': LSTMConfig(
            vocab_size=vocab_size,
            n_embd=512,
            n_layer=2,
            dropout=0.1
        ),
        'medium': LSTMConfig(
            vocab_size=vocab_size,
            n_embd=768,
            n_layer=3,
            dropout=0.1
        ),
        'large': LSTMConfig(
            vocab_size=vocab_size,
            n_embd=1024,
            n_layer=3,
            dropout=0.1
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    model = LSTMModel(config)
    
    return model