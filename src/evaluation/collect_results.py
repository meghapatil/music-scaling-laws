"""
Collect training results from all models for scaling analysis.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def power_law(x, a, alpha, c):
    """Power law function: L = a * N^(-alpha) + c"""
    return a * np.power(x, -alpha) + c


def collect_results(results_dir='results/scaling_study'):
    """Collect results from all trained models."""
    
    results_dir = Path(results_dir)
    
    all_results = []
    
    # Find all results.json files
    for results_file in results_dir.rglob('results.json'):
        with open(results_file, 'r') as f:
            data = json.load(f)
            
            # Extract key metrics
            result = {
                'model_type': data['model_type'],
                'model_size': data['model_size'],
                'num_params': data['num_params'],
                'final_train_loss': data['train_losses'][-1],
                'final_val_loss': data['val_losses'][-1],
                'best_val_loss': data['best_val_loss'],
                'avg_epoch_time': np.mean(data['epoch_times']),
            }
            
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by number of parameters
    df = df.sort_values('num_params')
    
    print("\n" + "="*60)
    print("COLLECTED RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    output_file = results_dir / 'all_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


def plot_scaling_laws(df, output_dir='results/scaling_study'):
    """Create scaling law plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate transformers and LSTMs
    transformers = df[df['model_type'] == 'transformer'].copy()
    lstms = df[df['model_type'] == 'lstm'].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Transformer Scaling
    ax = axes[0]
    if len(transformers) > 0:
        x = transformers['num_params'].values
        y = transformers['best_val_loss'].values
        
        # Fit power law
        try:
            params, _ = curve_fit(power_law, x, y, p0=[1, 0.1, 0.5], maxfev=10000)
            a, alpha, c = params
            
            # Plot data points
            ax.scatter(x, y, s=100, alpha=0.7, label='Data', color='blue')
            
            # Plot fitted curve
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_fit = power_law(x_fit, a, alpha, c)
            ax.plot(x_fit, y_fit, 'r--', label=f'Fit: L = {a:.3f}·N^(-{alpha:.3f}) + {c:.3f}', linewidth=2)
            
            # Annotate points
            for i, row in transformers.iterrows():
                ax.annotate(row['model_size'], 
                           (row['num_params'], row['best_val_loss']),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
            
            ax.set_title(f'Transformer Scaling Law\nα = {alpha:.4f}', fontsize=14, fontweight='bold')
            
        except Exception as e:
            print(f"Could not fit power law to transformers: {e}")
            ax.scatter(x, y, s=100, alpha=0.7, label='Data', color='blue')
            ax.set_title('Transformer Scaling', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 2: LSTM Scaling
    ax = axes[1]
    if len(lstms) > 0:
        x = lstms['num_params'].values
        y = lstms['best_val_loss'].values
        
        # Fit power law
        try:
            params, _ = curve_fit(power_law, x, y, p0=[1, 0.1, 0.5], maxfev=10000)
            a, alpha, c = params
            
            # Plot data points
            ax.scatter(x, y, s=100, alpha=0.7, label='Data', color='green')
            
            # Plot fitted curve
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_fit = power_law(x_fit, a, alpha, c)
            ax.plot(x_fit, y_fit, 'r--', label=f'Fit: L = {a:.3f}·N^(-{alpha:.3f}) + {c:.3f}', linewidth=2)
            
            # Annotate points
            for i, row in lstms.iterrows():
                ax.annotate(row['model_size'], 
                           (row['num_params'], row['best_val_loss']),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
            
            ax.set_title(f'LSTM Scaling Law\nα = {alpha:.4f}', fontsize=14, fontweight='bold')
            
        except Exception as e:
            print(f"Could not fit power law to LSTMs: {e}")
            ax.scatter(x, y, s=100, alpha=0.7, label='Data', color='green')
            ax.set_title('LSTM Scaling', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'scaling_laws.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Scaling plot saved to: {plot_file}")
    
    plt.show()
    
    # Combined comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(transformers) > 0:
        ax.scatter(transformers['num_params'], transformers['best_val_loss'], 
                  s=100, alpha=0.7, label='Transformer', color='blue', marker='o')
    
    if len(lstms) > 0:
        ax.scatter(lstms['num_params'], lstms['best_val_loss'], 
                  s=100, alpha=0.7, label='LSTM', color='green', marker='s')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Transformer vs LSTM Scaling Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = output_dir / 'architecture_comparison.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_file}")
    
    plt.show()


def plot_training_curves(df, output_dir='results/scaling_study'):
    """Plot training curves for all models."""
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot transformers
    ax = axes[0]
    transformers = df[df['model_type'] == 'transformer']
    for _, row in transformers.iterrows():
        # Load full results to get loss curves
        model_dir = output_dir / f"{row['model_type']}_{row['model_size']}"
        results_file = model_dir / 'results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            epochs = range(1, len(data['train_losses']) + 1)
            ax.plot(epochs, data['train_losses'], 
                   label=f"{row['model_size']} ({row['num_params']:,} params)",
                   marker='o', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Transformer Training Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot LSTMs
    ax = axes[1]
    lstms = df[df['model_type'] == 'lstm']
    for _, row in lstms.iterrows():
        model_dir = output_dir / f"{row['model_type']}_{row['model_size']}"
        results_file = model_dir / 'results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            epochs = range(1, len(data['train_losses']) + 1)
            ax.plot(epochs, data['train_losses'], 
                   label=f"{row['model_size']} ({row['num_params']:,} params)",
                   marker='o', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('LSTM Training Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    curves_file = output_dir / 'training_curves.png'
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {curves_file}")
    
    plt.show()


if __name__ == "__main__":
    # Collect results
    df = collect_results()
    
    # Create plots
    if len(df) > 0:
        plot_scaling_laws(df)
        plot_training_curves(df)
    else:
        print("\nNo results found yet. Train some models first!")