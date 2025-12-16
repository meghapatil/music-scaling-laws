# Music Scaling Laws: Training Language Models on Symbolic Music

**Author:** Megha Anant Patil (mp7464)  
**Course:** CS-GY 6923 Machine Learning  
**Institution:** NYU Tandon School of Engineering  
**Date:** December 15, 2025

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project investigates whether neural scaling laws transfer from natural language to symbolic music by training 9 models (5 transformers, 4 LSTMs) ranging from 800K to 50M parameters on ABC notation from the Lakh MIDI dataset.

**Key Finding:** Scaling laws did not emerge under experimental conditions due to insufficient training data (0.02-1.25 tokens/parameter vs. optimal 20+), demonstrating that model capacity must match data availability for scaling behavior to manifest.

## Results Summary

| Architecture | Scaling Exponent (α) | Best Performance | Outcome |
|-------------|---------------------|------------------|---------|
| Transformers | α ≈ 0.0001 (zero) | Small: 0.833 loss | No scaling |
| LSTMs | α ≈ -1.91 (negative) | Small: 0.833 loss | Degradation |

**Sample Generation:** 100% syntactic validity, limited musical coherence  
**Training Time:** ~30 CPU hours across all models  
**Data Used:** 1M tokens (4% of 25M available)

## Project Structure
```
music-scaling-project/
├── data/
│   ├── raw/                    # Original MIDI files (180GB, not in repo)
│   └── processed/              # Tokenized ABC notation
│       ├── train.npy           # 24.7M training tokens
│       ├── val.npy             # 261K validation tokens
│       ├── test.npy            # 90K test tokens
│       └── vocab.json          # 37-token vocabulary
├── src/
│   ├── convert_midi.py         # MIDI → ABC conversion
│   ├── convert_midi_safe.py    # Robust conversion with error handling
│   ├── tokenizer.py            # Character-level tokenization
│   ├── data_utils.py           # Data loading utilities
│   ├── run_preprocessing.py    # Complete preprocessing pipeline
│   ├── models/
│   │   ├── transformer.py      # GPT-style decoder
│   │   └── rnn.py              # Stacked LSTM
│   ├── training/
│   │   ├── dataset.py          # PyTorch dataset
│   │   ├── train.py            # Training script
│   │   └── train_all_9_models_1m.sh  # Batch training
│   ├── evaluation/
│   │   ├── collect_results.py  # Aggregate metrics
│   │   ├── generate_samples.py # Music generation
│   │   └── analyze_samples.py  # Sample analysis
│   └── visualization/
│       └── create_data_visualizations.py  # Data figures
├── results/
│   ├── scaling_study/          # Trained models & metrics
│   │   ├── transformer_tiny/
│   │   ├── transformer_small/
│   │   ├── transformer_medium/
│   │   ├── transformer_large/
│   │   ├── transformer_xl/
│   │   └── lstm_*/
│   ├── generated_samples/      # 30 ABC music samples
│   │   ├── tiny/   (10 samples)
│   │   ├── small/  (10 samples)
│   │   └── large/  (10 samples)
│   └── visualizations/         # All report figures
│       ├── sequence_length_distribution.png
│       ├── token_frequency.png
│       ├── dataset_split.png
│       └── scaling plots
├── configs/                    # Model configurations (if you have this)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 8GB RAM (minimum)
- 50GB disk space for dataset

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/music-scaling-project.git
cd music-scaling-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Data
```bash
# Download Lakh MIDI Dataset (116K files, ~180GB)
mkdir -p data/raw
cd data/raw
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz
cd ../..
```

### 2. Preprocess Data
```bash
# Complete pipeline: MIDI → ABC → Tokens
python src/run_preprocessing.py

# Or run steps individually:
python src/convert_midi.py --input_dir data/raw/lmd_matched --output_dir data/processed/abc
python src/tokenizer.py --input_dir data/processed/abc --output_dir data/processed
```

**Output:** 45,118 ABC files → 25M tokens (98%/1%/1% split)

### 3. Train Models

Train all 9 models:
```bash
bash src/training/train_all_9_models_1m.sh
```

Train individual model:
```bash
python src/training/train.py \
    --model_type transformer \
    --model_size small \
    --train_data data/processed/train_1m.npy \
    --val_data data/processed/val_1m.npy \
    --num_epochs 1 \
    --batch_size 128 \
    --output_dir results/scaling_study
```

**Training time:** ~30 hours total on Apple Silicon M-series CPU

### 4. Evaluate & Analyze
```bash
# Collect results and create scaling plots
python src/evaluation/collect_results.py

# Create data visualizations
python src/visualization/create_data_visualizations.py

# Analyze generated samples
python src/evaluation/analyze_samples.py
```

### 5. Generate Music
```bash
# Generate samples from trained models
python src/evaluation/generate_samples.py \
    --model_path results/scaling_study/transformer_small/best_model.pt \
    --vocab_path data/processed/vocab.json \
    --num_samples 10 \
    --temperature 0.8 \
    --output_dir results/generated_samples/small
```

**Listen to samples:**
1. Open generated `.abc` files
2. Visit https://abcjs.net/abcjs-editor.html
3. Paste ABC text to hear the music

## Model Configurations

### Transformers

| Model | Layers | d_model | Heads | Parameters | Val Loss |
|-------|--------|---------|-------|------------|----------|
| Tiny | 4 | 128 | 4 | 801K | 0.841 |
| Small | 6 | 192 | 6 | 2.7M | **0.833** |
| Medium | 8 | 256 | 8 | 6.3M | 0.836 |
| Large | 12 | 384 | 12 | 21M | 0.848 |
| XL | 16 | 512 | 16 | 50M | 0.880 |

### LSTMs

| Model | Layers | Hidden | Parameters | Val Loss |
|-------|--------|--------|------------|----------|
| Tiny | 2 | 256 | 1.1M | 0.859 |
| Small | 2 | 512 | 4.2M | **0.833** |
| Medium | 3 | 768 | 14M | 0.944 |
| Large | 3 | 1024 | 25M | 1.129 |

**All models:** dropout=0.1, block_size=64, AdamW optimizer, lr=3e-4

## Key Findings

### 1. Absence of Scaling Behavior

- **Transformers:** α ≈ 0.0001 (effectively zero)
- **LSTMs:** α ≈ -1.91 (negative)
- **Expected:** α ≈ 0.05-0.10 for proper scaling

### 2. Data-Compute Mismatch

| Model | Tokens/Param | Optimal | Shortfall |
|-------|--------------|---------|-----------|
| Tiny | 1.25 | 20+ | 16× |
| Small | 0.37 | 20+ | 54× |
| XL | 0.02 | 20+ | 1000× |

### 3. Training Dynamics

- **Training loss:** Decreases with size (0.816 → 0.726)
- **Validation loss:** Degrades with size (0.841 → 0.880)
- **Diagnosis:** Underfitting, not overfitting

### 4. Architecture Comparison

- **Small scale (< 5M):** Both architectures similar (~0.833)
- **Large scale (> 20M):** Transformers degrade 5%, LSTMs degrade 35%
- **Conclusion:** Transformers more robust to undertraining

### 5. Sample Quality

- **Syntactic validity:** 100% (all samples have proper ABC structure)
- **Musical coherence:** Limited (mostly rests with sparse notes)
- **Scaling effect:** No quality improvement Tiny → Large

## Implications

1. **For Music Generation:** With limited data (25M tokens), use 1-5M parameter models
2. **For Scaling Studies:** Need 20+ tokens/parameter (Chinchilla guidelines)
3. **For Practitioners:** Small well-trained models > Large undertrained models
4. **For Research:** Negative results valuable for experimental design

## Limitations

- Insufficient training compute (10-1000× below optimal)
- Small dataset (25M vs. typical billions for text)
- Single epoch evaluation
- CPU-only training
- No hyperparameter tuning
- Limited evaluation metrics

## Future Work

- Train on full 25M tokens for 5-10 epochs with GPU
- Extend dataset to 100M+ tokens
- Implement compute-optimal training (Chinchilla)
- Compare alternative tokenizations (REMI, piano roll)
- Human evaluation of generated music

## Citation
```bibtex
@misc{patil2025musicscaling,
  title={Music Scaling Laws: Training Language Models on Symbolic Music},
  author={Patil, Megha Anant},
  year={2025},
  institution={New York University Tandon School of Engineering},
  note={Negative results: Scaling laws require adequate data-compute balance}
}
```

## References

1. Kaplan et al. (2020). "Scaling Laws for Neural Language Models" - arXiv:2001.08361
2. Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" - arXiv:2203.15556
3. Raffel (2016). "Learning-Based Methods for Comparing Sequences"
4. Vaswani et al. (2017). "Attention Is All You Need"

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Dataset:** Lakh MIDI Dataset by Colin Raffel
- **Reference Code:** nanoGPT by Andrej Karpathy
- **ABC Notation:** ABC notation standard (abcnotation.com)

## Contact

**Megha Anant Patil**  
Email: mp7464@nyu.edu  
GitHub: [YOUR_USERNAME]

---

**Project Status:** ✅ Complete (December 2025)  
**Report:** 12 pages + appendix with honest analysis of negative results  
**Code:** Fully documented and reproducible
