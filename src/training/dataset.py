"""
Dataset utilities for loading tokenized music data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    """Dataset for tokenized music sequences."""
    
    def __init__(self, data_path, block_size):
        """
        Args:
            data_path: Path to .npy file with token IDs
            block_size: Sequence length for training
        """
        self.data = np.load(data_path)
        self.block_size = block_size
        
        print(f"Loaded dataset: {len(self.data):,} tokens")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get sequence of length block_size + 1
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Input and target (shifted by 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y


def create_dataloaders(train_path, val_path, block_size, batch_size, num_workers=0):
    """
    Create training and validation dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        block_size: Sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = MusicDataset(train_path, block_size)
    val_dataset = MusicDataset(val_path, block_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader