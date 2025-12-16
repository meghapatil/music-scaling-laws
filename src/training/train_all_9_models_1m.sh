#!/bin/bash
# Train ALL 9 models on 1M tokens - Fast and complete!

echo "======================================"
echo "TRAINING ALL 9 MODELS - 1M TOKENS"
echo "Started: $(date)"
echo "======================================"

mkdir -p results/scaling_study

# Model 1: Tiny Transformer
echo ""
echo "[1/9] Training Tiny Transformer..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type transformer \
    --model_size tiny \
    --num_epochs 1 \
    --batch_size 256 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Tiny Transformer complete: $(date)"

# Model 2: Small Transformer
echo ""
echo "[2/9] Training Small Transformer..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type transformer \
    --model_size small \
    --num_epochs 1 \
    --batch_size 128 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Small Transformer complete: $(date)"

# Model 3: Medium Transformer
echo ""
echo "[3/9] Training Medium Transformer..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type transformer \
    --model_size medium \
    --num_epochs 1 \
    --batch_size 64 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Medium Transformer complete: $(date)"

# Model 4: Large Transformer
echo ""
echo "[4/9] Training Large Transformer..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type transformer \
    --model_size large \
    --num_epochs 1 \
    --batch_size 32 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Large Transformer complete: $(date)"

# Model 5: XL Transformer
echo ""
echo "[5/9] Training XL Transformer..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type transformer \
    --model_size xl \
    --num_epochs 1 \
    --batch_size 16 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ XL Transformer complete: $(date)"

# Model 6: Tiny LSTM
echo ""
echo "[6/9] Training Tiny LSTM..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type lstm \
    --model_size tiny \
    --num_epochs 1 \
    --batch_size 256 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Tiny LSTM complete: $(date)"

# Model 7: Small LSTM
echo ""
echo "[7/9] Training Small LSTM..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type lstm \
    --model_size small \
    --num_epochs 1 \
    --batch_size 128 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Small LSTM complete: $(date)"

# Model 8: Medium LSTM
echo ""
echo "[8/9] Training Medium LSTM..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type lstm \
    --model_size medium \
    --num_epochs 1 \
    --batch_size 64 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Medium LSTM complete: $(date)"

# Model 9: Large LSTM
echo ""
echo "[9/9] Training Large LSTM..."
echo "Started: $(date)"
python src/training/train.py \
    --model_type lstm \
    --model_size large \
    --num_epochs 1 \
    --batch_size 32 \
    --block_size 64 \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --output_dir results/scaling_study
echo "✓ Large LSTM complete: $(date)"

echo ""
echo "======================================"
echo "ALL 9 MODELS COMPLETE!"
echo "Finished: $(date)"
echo "======================================"
