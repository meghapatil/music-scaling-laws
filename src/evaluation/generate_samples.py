import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
sys.path.append('src/models')

# Model size configurations
MODEL_CONFIGS = {
    'transformer': {
        'tiny': {'n_layer': 4, 'n_head': 4, 'n_embd': 128},
        'small': {'n_layer': 6, 'n_head': 6, 'n_embd': 192},
        'medium': {'n_layer': 8, 'n_head': 8, 'n_embd': 256},
        'large': {'n_layer': 12, 'n_head': 12, 'n_embd': 384},
        'xl': {'n_layer': 16, 'n_head': 16, 'n_embd': 512}
    }
}

class Config:
    """Configuration object for model"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_vocab(vocab_path):
    """Load vocabulary"""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['char_to_id']

def generate_sample(model, start_tokens, max_length, temperature, device, id_to_char):
    """Generate a single sample from the model"""
    model.eval()
    
    tokens = start_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Limit context to model's block size
            context = tokens if tokens.size(1) <= 64 else tokens[:, -64:]
            
            # Get model predictions - handle both (logits,) and (logits, loss) returns
            output = model(context)
            if isinstance(output, tuple):
                logits = output[0]  # Model returns (logits, loss)
            else:
                logits = output  # Model returns just logits
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if we've generated enough
            if tokens.size(1) >= start_tokens.size(1) + max_length:
                break
    
    # Convert tokens to text
    text = ''.join([id_to_char.get(t.item(), '') for t in tokens[0]])
    return text

def create_abc_header(number, title="Generated", meter="4/4", key="C"):
    """Create valid ABC header"""
    return f"X:{number}\nT:{title}\nM:{meter}\nL:1/8\nK:{key}\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocab.json')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default='results/generated_samples')
    args = parser.parse_args()
    
    # Load vocabulary
    char_to_id = load_vocab(args.vocab_path)
    id_to_char = {v: k for k, v in char_to_id.items()}
    vocab_size = len(char_to_id)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Loading model from: {args.model_path}")
    
    # Load model
    device = torch.device('cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model type and size from config
    saved_config = checkpoint.get('config', checkpoint.get('model_config', {}))
    model_type = saved_config.get('model_type', 'transformer')
    model_size = saved_config.get('model_size', 'tiny')
    
    print(f"Model type: {model_type}")
    print(f"Model size: {model_size}")
    
    # Get architecture config
    arch_config = MODEL_CONFIGS[model_type][model_size]
    block_size = saved_config.get('block_size', 64)
    dropout = saved_config.get('dropout', 0.1)
    
    print(f"Architecture: {arch_config}")
    
    # Create config object
    from transformer import Transformer
    
    config = Config(
        n_vocab=vocab_size,
        vocab_size=vocab_size,
        n_layer=arch_config['n_layer'],
        n_head=arch_config['n_head'],
        n_embd=arch_config['n_embd'],
        block_size=block_size,
        dropout=dropout,
        bias=False
    )
    
    model = Transformer(config)
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"Generating {args.num_samples} samples...\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    samples = []
    valid_count = 0
    
    for i in range(args.num_samples):
        # Create ABC header
        meters = ["4/4", "3/4", "6/8", "2/4"]
        keys = ["C", "G", "D", "A", "F", "Bb"]
        
        header = create_abc_header(
            number=i+1,
            title=f"AI_Song_{i+1}",
            meter=np.random.choice(meters),
            key=np.random.choice(keys)
        )
        
        # Convert header to tokens
        start_tokens = torch.tensor(
            [[char_to_id.get(c, char_to_id.get(' ', 0)) for c in header]], 
            device=device
        )
        
        # Generate
        try:
            sample_text = generate_sample(
                model, start_tokens, args.max_length, args.temperature, device, id_to_char
            )
            
            samples.append(sample_text)
            
            # Check validity
            has_header = 'X:' in sample_text
            has_key = 'K:' in sample_text
            has_barlines = '|' in sample_text
            has_notes = any(c in sample_text for c in 'ABCDEFGabcdefg')
            
            is_valid = has_header and has_key and has_barlines and has_notes
            
            if is_valid:
                valid_count += 1
            
            # Save individual sample
            with open(output_dir / f'sample_{i+1}.abc', 'w') as f:
                f.write(sample_text)
            
            # Show preview
            preview = sample_text[40:120].replace('\n', ' ')
            status = '✓' if is_valid else '✗'
            print(f"[{status}] Sample {i+1}/{args.num_samples}: {len(sample_text)} chars")
            print(f"    {preview}...")
            
        except Exception as e:
            print(f"[✗] Sample {i+1}/{args.num_samples}: Error - {e}")
            import traceback
            traceback.print_exc()
            samples.append(f"# Error generating sample {i+1}\n")
    
    # Save all samples
    with open(output_dir / 'all_samples.abc', 'w') as f:
        f.write('\n\n'.join(samples))
    
    # Save statistics
    stats = {
        'num_samples': args.num_samples,
        'valid_samples': valid_count,
        'validity_rate': valid_count / args.num_samples if args.num_samples > 0 else 0,
        'model_path': str(args.model_path),
        'temperature': args.temperature,
        'model_type': model_type,
        'model_size': model_size
    }
    
    with open(output_dir / 'generation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Generation Complete!")
    print(f"{'='*60}")
    print(f"Valid samples: {valid_count}/{args.num_samples} ({valid_count/args.num_samples*100:.1f}%)")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
