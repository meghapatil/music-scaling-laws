"""
Data utilities for cleaning, filtering, and splitting ABC files.
"""

import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def clean_and_filter_abc_files(abc_dir, min_length=100, max_length=10000):
    """
    Filter ABC files based on quality criteria.
    
    Args:
        abc_dir: Directory containing ABC files
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        list: Paths to valid ABC files
    """
    abc_dir = Path(abc_dir)
    abc_files = list(abc_dir.glob("*.abc"))
    
    print(f"Found {len(abc_files)} ABC files")
    print("Filtering files...")
    
    valid_files = []
    stats = {
        'too_short': 0,
        'too_long': 0,
        'no_notes': 0,
        'valid': 0
    }
    
    for abc_file in tqdm(abc_files, desc="Filtering"):
        try:
            with open(abc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply filters
            if len(content) < min_length:
                stats['too_short'] += 1
                continue
            
            if len(content) > max_length:
                stats['too_long'] += 1
                continue
            
            # Check if file contains actual notes
            if not any(note in content for note in 'CDEFGABcdefgab'):
                stats['no_notes'] += 1
                continue
            
            valid_files.append(abc_file)
            stats['valid'] += 1
            
        except Exception as e:
            print(f"Error reading {abc_file}: {e}")
            continue
    
    print("\nFiltering Results:")
    print(f"  Valid: {stats['valid']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"  Too long: {stats['too_long']}")
    print(f"  No notes: {stats['no_notes']}")
    print(f"  Keep rate: {100*stats['valid']/len(abc_files):.1f}%")
    
    return valid_files


def create_splits(abc_files, train_ratio=0.98, val_ratio=0.01, test_ratio=0.01, seed=42):
    """
    Split ABC files into train/validation/test sets.
    
    Args:
        abc_files: List of ABC file paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        tuple: (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Shuffle with fixed seed
    random.seed(seed)
    files = abc_files.copy()
    random.shuffle(files)
    
    # Calculate split points
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    print("\nData Split:")
    print(f"  Train: {len(train_files)} files ({100*len(train_files)/n:.1f}%)")
    print(f"  Val:   {len(val_files)} files ({100*len(val_files)/n:.1f}%)")
    print(f"  Test:  {len(test_files)} files ({100*len(test_files)/n:.1f}%)")
    
    return train_files, val_files, test_files


def load_and_concatenate_files(file_list, separator="\n\n"):
    """
    Load and concatenate all ABC files.
    
    Args:
        file_list: List of file paths
        separator: String to insert between files
        
    Returns:
        str: Concatenated text
    """
    all_text = []
    
    for file_path in tqdm(file_list, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            all_text.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    full_text = separator.join(all_text)
    print(f"Loaded {len(all_text)} files, total length: {len(full_text):,} characters")
    
    return full_text


def tokenize_and_save(text, tokenizer, output_path):
    """
    Tokenize text and save as numpy array.
    
    Args:
        text: Input text
        tokenizer: Tokenizer object
        output_path: Where to save tokens
        
    Returns:
        np.array: Token IDs
    """
    print(f"Tokenizing {len(text):,} characters...")
    tokens = tokenizer.encode(text)
    tokens_array = np.array(tokens, dtype=np.uint32)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tokens_array)
    
    print(f"Saved {len(tokens):,} tokens to {output_path}")
    
    return tokens_array


def compute_dataset_statistics(train_tokens, val_tokens, test_tokens, vocab_size):
    """
    Compute and display dataset statistics.
    
    Args:
        train_tokens: Training token array
        val_tokens: Validation token array
        test_tokens: Test token array
        vocab_size: Size of vocabulary
        
    Returns:
        dict: Statistics dictionary
    """
    total = len(train_tokens) + len(val_tokens) + len(test_tokens)
    
    stats = {
        'vocab_size': vocab_size,
        'total_tokens': total,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'test_tokens': len(test_tokens),
        'train_pct': 100 * len(train_tokens) / total,
        'val_pct': 100 * len(val_tokens) / total,
        'test_pct': 100 * len(test_tokens) / total,
    }
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"  Train: {stats['train_tokens']:,} ({stats['train_pct']:.2f}%)")
    print(f"  Val:   {stats['val_tokens']:,} ({stats['val_pct']:.2f}%)")
    print(f"  Test:  {stats['test_tokens']:,} ({stats['test_pct']:.2f}%)")
    print("="*50)
    
    return stats


def save_statistics(stats, output_path):
    """Save statistics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {output_path}")