"""
Visualization utilities for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predicted vs. Ground Truth Quality Scores",
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    plcc: Optional[float] = None,
    srcc: Optional[float] = None
) -> None:
    """
    Plot scatter plot of predictions vs. targets.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
        title: Plot title
        save_path: Path to save figure
        show_metrics: Whether to show correlation metrics
        plcc: PLCC value to display
        srcc: SRCC value to display
    """
    plt.figure(figsize=(8, 8))
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Fit line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', linewidth=2, alpha=0.8, label='Fitted line')
    
    # Labels and title
    plt.xlabel('Ground Truth Quality Score', fontsize=14)
    plt.ylabel('Predicted Quality Score', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add metrics text
    if show_metrics and (plcc is not None or srcc is not None):
        text = ""
        if plcc is not None:
            text += f"PLCC: {plcc:.4f}\n"
        if srcc is not None:
            text += f"SRCC: {srcc:.4f}"
        
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss curves).
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].set_title('Training and Validation Loss', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history:
        axes[1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Learning Rate', fontsize=14)
        axes[1].set_title('Learning Rate Schedule', fontsize=16)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_metric_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['plcc', 'srcc', 'rmse', 'mae'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of metrics across different models/methods.
    
    Args:
        results: Dictionary of method_name -> metrics_dict
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    method_names = list(results.keys())
    
    for idx, metric in enumerate(metrics):
        values = [results[method][metric] for method in method_names]
        
        # Bar plot
        bars = axes[idx].bar(range(len(method_names)), values, alpha=0.7)
        axes[idx].set_xticks(range(len(method_names)))
        axes[idx].set_xticklabels(method_names, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.upper(), fontsize=14)
        axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=16)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Color bars by value (green for better, red for worse)
        if metric in ['plcc', 'srcc']:
            # Higher is better
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
        else:
            # Lower is better (RMSE, MAE)
            colors = ['green' if v < 0.1 else 'orange' if v < 0.2 else 'red' for v in values]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metric comparison plot saved to {save_path}")
    
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot error distribution (histogram and box plot).
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Compute errors
    errors = predictions.flatten() - targets.flatten()
    abs_errors = np.abs(errors)
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean error: {errors.mean():.4f}')
    axes[0].set_xlabel('Prediction Error', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].set_title('Error Distribution', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    box_data = [errors, abs_errors]
    axes[1].boxplot(box_data, labels=['Error', 'Absolute Error'])
    axes[1].set_ylabel('Value', fontsize=14)
    axes[1].set_title('Error Statistics', fontsize=16)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Mean Error: {errors.mean():.4f}\n"
    stats_text += f"Std Error: {errors.std():.4f}\n"
    stats_text += f"MAE: {abs_errors.mean():.4f}"
    
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    
    plt.close()


def plot_quality_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of quality scores (predicted vs. ground truth).
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Histograms
    plt.hist(targets, bins=30, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
    plt.hist(predictions, bins=30, alpha=0.5, label='Predictions', color='red', edgecolor='black')
    
    plt.xlabel('Quality Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Quality Score Distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quality distribution plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix_binned(
    predictions: np.ndarray,
    targets: np.ndarray,
    bins: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for binned quality scores.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
        bins: Number of bins for quality categories
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Bin the scores
    pred_bins = np.digitize(predictions, np.linspace(0, 1, bins+1)[:-1])
    target_bins = np.digitize(targets, np.linspace(0, 1, bins+1)[:-1])
    
    # Compute confusion matrix
    cm = confusion_matrix(target_bins, pred_bins)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(1, bins+1), yticklabels=range(1, bins+1))
    plt.xlabel('Predicted Quality Bin', fontsize=14)
    plt.ylabel('True Quality Bin', fontsize=14)
    plt.title(f'Confusion Matrix (Quality Binned into {bins} Categories)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()
