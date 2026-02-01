"""
Evaluation script for IQA model.
"""

import torch
import argparse
from pathlib import Path

from evaluation import IQAEvaluator
from data import create_dataloaders
from models.deep_learning import get_model
from models.utils import load_checkpoint


def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str = 'evaluation_results',
    device: str = 'cuda',
    batch_size: int = 32,
    model_type: str = 'lightweight'
):
    """
    Evaluate IQA model on test set.
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Path to data directory
        output_dir: Output directory for results
        device: Device for evaluation
        batch_size: Batch size
        model_type: Type of model
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluation Settings:")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading model...")
    model = get_model(model_type=model_type)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    # Create data loader
    print("\nCreating data loaders...")
    data_loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    test_loader = data_loaders['test']
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    evaluator = IQAEvaluator(
        model=model,
        device=device,
        output_dir=output_dir
    )
    
    # Evaluate
    metrics = evaluator.evaluate(test_loader, save_predictions=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        from evaluation.visualization import (
            plot_scatter,
            plot_error_distribution,
            plot_quality_distribution
        )
        import numpy as np
        
        predictions = np.array(evaluator.results['predictions'])
        targets = np.array(evaluator.results['targets'])
        
        # Scatter plot
        plot_scatter(
            predictions,
            targets,
            save_path=str(output_path / 'scatter_plot.png'),
            plcc=metrics['plcc'],
            srcc=metrics['srcc']
        )
        
        # Error distribution
        plot_error_distribution(
            predictions,
            targets,
            save_path=str(output_path / 'error_distribution.png')
        )
        
        # Quality distribution
        plot_quality_distribution(
            predictions,
            targets,
            save_path=str(output_path / 'quality_distribution.png')
        )
        
        print("Visualizations saved!")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    print("\nEvaluation complete!")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate IQA model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightweight',
        choices=['lightweight', 'full'],
        help='Model type'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        model_type=args.model_type
    )


if __name__ == '__main__':
    main()
