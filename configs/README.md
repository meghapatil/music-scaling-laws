# Model Configuration Files

This directory contains JSON configuration files for all 9 models used in the scaling study.

## Transformer Models

- `transformer_tiny.json` - 4 layers, 128 dim, 800K params
- `transformer_small.json` - 6 layers, 192 dim, 5M params
- `transformer_medium.json` - 8 layers, 256 dim, 20M params
- `transformer_large.json` - 12 layers, 384 dim, 50M params
- `transformer_xl.json` - 16 layers, 512 dim, 100M params

## LSTM Models

- `lstm_tiny.json` - 2 layers, 256 hidden, 1M params
- `lstm_small.json` - 2 layers, 512 hidden, 5M params
- `lstm_medium.json` - 3 layers, 768 hidden, 20M params
- `lstm_large.json` - 3 layers, 1024 hidden, 50M params

## Configuration Parameters

### Common Parameters

- `vocab_size`: 37 (ABC notation character set)
- `block_size`: 64 (sequence length)
- `dropout`: 0.1
- `learning_rate`: 0.0003
- `weight_decay`: 0.01
- `grad_clip`: 1.0

### Transformer Specific

- `n_layer`: Number of transformer blocks
- `n_head`: Number of attention heads
- `n_embd`: Embedding/hidden dimension
- `bias`: Whether to use bias in linear layers

### LSTM Specific

- `n_layer`: Number of LSTM layers
- `hidden_size`: Hidden state dimension
- `n_embd`: Embedding dimension (fixed at 256)

## Usage

Load a config in Python:
```python
import json

with open('configs/transformer_tiny.json', 'r') as f:
    config = json.load(f)

# Use config
model = TransformerModel(**config)
```

Or from command line:
```bash
python src/training/train.py --config configs/transformer_tiny.json
```

## Master Config

- `scaling_study_config.json` - Master configuration listing all models and experiment settings
- `data_config.json` - Data preprocessing and path configuration
