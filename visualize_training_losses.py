"""
Visualization Script for Training and Validation Loss Plots
============================================================

This script loads training metrics from JSON files and creates
plots showing training vs validation loss curves.

Usage:
    python visualize_training_losses.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(metrics_dir: Path = Path("training_metrics")):
    """Load all metrics files from the training_metrics directory."""
    metrics = {}
    for file in metrics_dir.glob("metrics_*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            repo_name = data.get("repository", file.stem.replace("metrics_", ""))
            metrics[repo_name] = data
    return metrics


def plot_mse_comparison(metrics: dict, output_path: str = "training_plots/mse_comparison.png"):
    """Create bar chart comparing train/val MSE across repositories."""
    repos = list(metrics.keys())
    train_mse = [m["metrics"]["train_mse"] for m in metrics.values()]
    val_mse = [m["metrics"]["val_mse"] for m in metrics.values()]
    
    x = np.arange(len(repos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_mse, width, label='Train MSE', color='steelblue')
    bars2 = ax.bar(x + width/2, val_mse, width, label='Validation MSE', color='coral')
    
    ax.set_xlabel('Repository')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Training vs Validation MSE by Repository')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('__', '/') for r in repos], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_r2_comparison(metrics: dict, output_path: str = "training_plots/r2_comparison.png"):
    """Create bar chart comparing train/val R² across repositories."""
    repos = list(metrics.keys())
    train_r2 = [m["metrics"]["train_r2"] for m in metrics.values()]
    val_r2 = [m["metrics"]["val_r2"] for m in metrics.values()]
    
    x = np.arange(len(repos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', color='forestgreen')
    bars2 = ax.bar(x + width/2, val_r2, width, label='Validation R²', color='orange')
    
    ax.set_xlabel('Repository')
    ax.set_ylabel('R² Score')
    ax.set_title('Training vs Validation R² by Repository')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('__', '/') for r in repos], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_train_val_loss_curves(
    train_losses: list,
    val_losses: list,
    repository_name: str = "Repository",
    output_path: str = "training_plots/loss_curves.png"
):
    """
    Plot training and validation loss curves over epochs.
    
    Args:
        train_losses: List of training loss values per epoch.
        val_losses: List of validation loss values per epoch.
        repository_name: Name of the repository for the title.
        output_path: Path to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'{repository_name} - Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE) - Log Scale')
    ax2.set_title(f'{repository_name} - Loss Curves (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_sample_loss_curves():
    """Generate sample loss curves for demonstration."""
    # Simulate typical training dynamics
    np.random.seed(42)
    epochs = 100
    
    # Training loss: starts high, decreases with noise
    train_loss = 0.8 * np.exp(-np.arange(epochs) / 30) + 0.05 + np.random.randn(epochs) * 0.02
    train_loss = np.maximum(train_loss, 0.01)
    
    # Validation loss: decreases then plateaus (slight overfitting)
    val_loss = 0.9 * np.exp(-np.arange(epochs) / 25) + 0.15 + np.random.randn(epochs) * 0.03
    val_loss = np.maximum(val_loss, 0.05)
    val_loss[50:] += 0.02 * np.arange(50) / 50  # slight increase (overfitting)
    
    return train_loss.tolist(), val_loss.tolist()


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Training Loss Visualization Script")
    print("=" * 60)
    
    # Load existing metrics
    metrics_dir = Path("training_metrics")
    if metrics_dir.exists():
        metrics = load_metrics(metrics_dir)
        if metrics:
            print(f"\nLoaded metrics for {len(metrics)} repositories")
            plot_mse_comparison(metrics)
            plot_r2_comparison(metrics)
        else:
            print("No metrics files found in training_metrics/")
    
    # Generate sample loss curves for demonstration
    print("\nGenerating sample training/validation loss curves...")
    train_losses, val_losses = generate_sample_loss_curves()
    plot_train_val_loss_curves(
        train_losses,
        val_losses,
        repository_name="Sample Repository",
        output_path="training_plots/sample_loss_curves.png"
    )
    
    print("\n" + "=" * 60)
    print("All plots saved to training_plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
