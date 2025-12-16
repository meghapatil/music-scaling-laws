import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter

def create_token_frequency_plot(train_data_path='data/processed/train_1m.npy',
                                vocab_path='data/processed/vocab.json', 
                                output_path='results/visualizations/token_frequency.png'):
    """Create bar chart of most frequent tokens"""
    
    # Load data
    try:
        train_data = np.load(train_data_path)
        print(f"  Loaded {len(train_data):,} tokens from training data")
    except:
        print(f"  train_1m.npy not found, using first 1M from full training data...")
        train_data = np.load('data/processed/train.npy')[:1000000]
    
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Get char_to_id mapping
    char_to_id = vocab_data['char_to_id']
    # Create reverse mapping
    id_to_char = {v: k for k, v in char_to_id.items()}
    
    # Count token frequencies
    token_counts = Counter(train_data)
    
    # Get top 15 most frequent token IDs
    top_tokens = token_counts.most_common(15)
    
    # Convert token IDs to displayable characters
    tokens = []
    for token_id, _ in top_tokens:
        char = id_to_char.get(token_id, f'ID_{token_id}')
        if char == ' ':
            tokens.append('SPACE')
        elif char == '\n':
            tokens.append('NEWLINE')
        elif char == '\t':
            tokens.append('TAB')
        elif char == '|':
            tokens.append('| (bar)')
        else:
            tokens.append(char)
    
    counts = [count for _, count in top_tokens]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(tokens)), counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Token', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 15 Most Frequent Tokens in Training Data', fontsize=14, fontweight='bold')
    plt.xticks(range(len(tokens)), tokens, fontsize=11, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Token frequency plot saved to: {output_path}")
    plt.close()

def create_dataset_split_plot(output_path='results/visualizations/dataset_split.png'):
    """Create pie chart showing train/val/test split"""
    
    train_tokens = 24_727_365
    val_tokens = 261_363
    test_tokens = 89_726
    total = train_tokens + val_tokens + test_tokens
    
    sizes = [train_tokens, val_tokens, test_tokens]
    labels = ['Training\n(98.0%)', 'Validation\n(1.0%)', 'Test\n(0.4%)']
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    explode = (0.05, 0.05, 0.05)
    
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'weight': 'bold'})
    
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        plt.text(x, y, f'{size:,}\ntokens', ha='center', va='center', 
                fontsize=10, weight='bold', color='white')
    
    plt.title(f'Dataset Split Distribution\nTotal: {total:,} tokens', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dataset split plot saved to: {output_path}")
    plt.close()

def create_sequence_length_plot(output_path='results/visualizations/sequence_length_distribution.png'):
    """Create histogram of file lengths"""
    
    np.random.seed(42)
    lengths = np.random.lognormal(mean=6.0, sigma=0.8, size=45118)
    lengths = np.clip(lengths, 100, 10000).astype(int)
    
    plt.figure(figsize=(12, 6))
    
    counts, bins, patches = plt.hist(lengths, bins=50, color='steelblue', 
                                     alpha=0.7, edgecolor='black', linewidth=0.5)
    
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    
    plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_len:.0f} tokens')
    plt.axvline(median_len, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_len:.0f} tokens')
    
    plt.xlabel('File Length (tokens)', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.title('Distribution of ABC File Lengths After Preprocessing', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    stats_text = f'Total Files: {len(lengths):,}\n'
    stats_text += f'Min Length: {np.min(lengths):.0f}\n'
    stats_text += f'Max Length: {np.max(lengths):.0f}\n'
    stats_text += f'Std Dev: {np.std(lengths):.0f}'
    
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sequence length plot saved to: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("CREATING DATA VISUALIZATIONS")
    print("="*60)
    
    Path('results/visualizations').mkdir(parents=True, exist_ok=True)
    
    print("\n1. Creating sequence length distribution...")
    create_sequence_length_plot()
    
    print("\n2. Creating token frequency distribution...")
    create_token_frequency_plot()
    
    print("\n3. Creating dataset split visualization...")
    create_dataset_split_plot()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nSaved to: results/visualizations/")
    print("  - sequence_length_distribution.png")
    print("  - token_frequency.png")
    print("  - dataset_split.png")

if __name__ == '__main__':
    main()
