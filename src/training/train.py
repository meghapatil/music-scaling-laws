"""
Training script for music language models.
Supports both transformer and LSTM architectures.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import create_model as create_transformer
from models.rnn import create_lstm_model
from training.dataset import create_dataloaders


def train_epoch(model, train_loader, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = model(x, targets=y)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, targets=y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.model_type}_{args.model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocab size
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
        if 'vocab_size' in vocab_data:
            vocab_size = vocab_data['vocab_size']
        elif 'char_to_id' in vocab_data:
            vocab_size = len(vocab_data['char_to_id'])
        elif 'token_to_id' in vocab_data:
            vocab_size = len(vocab_data['token_to_id'])
        else:
            raise ValueError("Cannot determine vocab size from vocab file")
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print(f"Creating {args.model_type} model (size: {args.model_size})...")
    if args.model_type == 'transformer':
        model = create_transformer(args.model_size, vocab_size, block_size=args.block_size)
    elif args.model_type == 'lstm':
        model = create_lstm_model(args.model_size, vocab_size)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        args.train_data,
        args.val_data,
        args.block_size,
        args.batch_size,
        num_workers=args.num_workers
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epoch(s)...")
    results = {
        'model_type': args.model_type,
        'model_size': args.model_size,
        'num_params': model.get_num_params(),
        'train_losses': [],
        'val_losses': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        # Log results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['epoch_times'].append(epoch_time)
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final results
    results['best_val_loss'] = best_val_loss
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train music language model")
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'lstm'],
                        help='Model architecture')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large', 'xl'],
                        help='Model size')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/processed/train.npy',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/processed/val.npy',
                        help='Path to validation data')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocab.json',
                        help='Path to vocabulary file')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    train(args)