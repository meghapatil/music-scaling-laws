"""
Tokenization schemes for ABC notation
Supports character-level and note-level tokenization.
"""

import re
import json
from pathlib import Path
from collections import Counter


class CharacterTokenizer:
    """Simple character-level tokenizer for ABC notation."""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
        
        print(f"Built character vocabulary: {self.vocab_size} tokens")
        return self.char_to_id
    
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return ''.join([self.id_to_char.get(tid, '?') for tid in token_ids])
    
    def save(self, path):
        """Save vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path):
        """Load vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self.vocab_size = data['vocab_size']


class NoteLevelTokenizer:
    """Note-level tokenizer that parses ABC musical elements."""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
    
    def tokenize_text(self, abc_text):
        """Split ABC text into musical tokens."""
        # Pattern matches: notes, accidentals, durations, bar lines, etc.
        pattern = r'[A-Ga-g][\'`,]*[0-9]*/?[0-9]*|[_^=]|[|:\[\]()]|\s+|.'
        tokens = re.findall(pattern, abc_text)
        # Remove pure whitespace but keep meaningful tokens
        tokens = [t for t in tokens if t.strip() or t in '|\n']
        return tokens
    
    def build_vocab(self, text):
        """Build vocabulary from ABC text."""
        tokens = self.tokenize_text(text)
        unique_tokens = sorted(set(tokens))
        
        self.token_to_id = {tok: i for i, tok in enumerate(unique_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.vocab_size = len(unique_tokens)
        
        print(f"Built note-level vocabulary: {self.vocab_size} tokens")
        
        # Show some example tokens
        print("Example tokens:", unique_tokens[:20])
        
        return self.token_to_id
    
    def encode(self, text):
        """Convert ABC text to token IDs."""
        tokens = self.tokenize_text(text)
        return [self.token_to_id.get(tok, 0) for tok in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs back to ABC text."""
        return ''.join([self.id_to_token.get(tid, '?') for tid in token_ids])
    
    def save(self, path):
        """Save vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path):
        """Load vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.vocab_size = data['vocab_size']


def get_tokenizer(tokenizer_type='character'):
    """
    Factory function to get the appropriate tokenizer.
    
    Args:
        tokenizer_type: 'character' or 'note'
    """
    if tokenizer_type == 'character':
        return CharacterTokenizer()
    elif tokenizer_type == 'note':
        return NoteLevelTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")