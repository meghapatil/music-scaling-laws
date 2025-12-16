"""
Main preprocessing pipeline for music scaling project.
Run this script to process your MIDI data from start to finish.

Usage:
    python src/run_preprocessing.py --tokenizer character
    python src/run_preprocessing.py --tokenizer note --max_files 1000
"""

import argparse
from pathlib import Path

from convert_midi import batch_convert_midi_to_abc
from tokenizer import get_tokenizer
from data_utils import (
    clean_and_filter_abc_files,
    create_splits,
    load_and_concatenate_files,
    tokenize_and_save,
    compute_dataset_statistics,
    save_statistics
)


def main(args):
    """Run the complete preprocessing pipeline."""
    
    print("="*70)
    print("MUSIC SCALING PROJECT - DATA PREPROCESSING")
    print("="*70)
    
    # ===== STEP 1: Convert MIDI to ABC =====
    print("\n[STEP 1] Converting MIDI to ABC notation...")
    if args.skip_conversion:
        print("  Skipping conversion (using existing ABC files)")
    else:
        successful, failed, _ = batch_convert_midi_to_abc(
            midi_dir=args.midi_dir,
            output_dir=args.abc_dir,
            max_files=args.max_files
        )
        
        if successful == 0:
            print("ERROR: No files were converted successfully!")
            return
    
    # ===== STEP 2: Filter ABC files =====
    print("\n[STEP 2] Filtering ABC files...")
    valid_files = clean_and_filter_abc_files(
        abc_dir=args.abc_dir,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    if len(valid_files) == 0:
        print("ERROR: No valid ABC files found!")
        return
    
    # ===== STEP 3: Create train/val/test splits =====
    print("\n[STEP 3] Creating data splits...")
    train_files, val_files, test_files = create_splits(
        valid_files,
        train_ratio=0.98,
        val_ratio=0.01,
        test_ratio=0.01
    )
    
    # ===== STEP 4: Initialize tokenizer =====
    print(f"\n[STEP 4] Initializing {args.tokenizer} tokenizer...")
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Build vocabulary on training data only
    print("  Loading training files to build vocabulary...")
    train_text = load_and_concatenate_files(train_files)
    tokenizer.build_vocab(train_text)
    
    # Save vocabulary
    vocab_path = Path(args.output_dir) / "vocab.json"
    tokenizer.save(vocab_path)
    print(f"  Vocabulary saved to {vocab_path}")
    
    # ===== STEP 5: Tokenize and save all splits =====
    print("\n[STEP 5] Tokenizing all splits...")
    
    # Training set (already loaded)
    train_tokens = tokenize_and_save(
        train_text,
        tokenizer,
        Path(args.output_dir) / "train.npy"
    )
    
    # Validation set
    print("\nProcessing validation set...")
    val_text = load_and_concatenate_files(val_files)
    val_tokens = tokenize_and_save(
        val_text,
        tokenizer,
        Path(args.output_dir) / "val.npy"
    )
    
    # Test set
    print("\nProcessing test set...")
    test_text = load_and_concatenate_files(test_files)
    test_tokens = tokenize_and_save(
        test_text,
        tokenizer,
        Path(args.output_dir) / "test.npy"
    )
    
    # ===== STEP 6: Compute and save statistics =====
    print("\n[STEP 6] Computing dataset statistics...")
    stats = compute_dataset_statistics(
        train_tokens,
        val_tokens,
        test_tokens,
        tokenizer.vocab_size
    )
    
    stats_path = Path(args.output_dir) / "dataset_stats.json"
    save_statistics(stats, stats_path)
    
    # ===== DONE =====
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"Processed data saved to: {args.output_dir}")
    print(f"Ready to train models with {stats['train_tokens']:,} training tokens")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI data for music scaling project")
    
    # Input/output paths
    parser.add_argument("--midi_dir", type=str, default="data/raw_midi",
                        help="Directory containing MIDI files")
    parser.add_argument("--abc_dir", type=str, default="data/abc_notation",
                        help="Directory to save ABC files")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    
    # Processing options
    parser.add_argument("--tokenizer", type=str, default="character",
                        choices=["character", "note"],
                        help="Tokenization scheme to use")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of MIDI files to process (None = all)")
    parser.add_argument("--min_length", type=int, default=100,
                        help="Minimum ABC file length in characters")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Maximum ABC file length in characters")
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip MIDI conversion (use existing ABC files)")
    
    args = parser.parse_args()
    
    main(args)
    