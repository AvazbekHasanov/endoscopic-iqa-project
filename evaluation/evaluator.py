"""
Evaluator class for comprehensive model evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import time

from .metrics import (
    compute_plcc, compute_srcc, compute_rmse, compute_mae,
    compute_all_metrics, compute_confidence_interval
)


class IQAEvaluator:
    """
    Evaluator for IQA models.
    Handles model evaluation, metric computation, and result logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        output_dir: str = 'evaluation_results'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: IQA model
            device: Device for evaluation
            output_dir: Directory for saving results
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def evaluate(
        self,
        data_loader: DataLoader,
        save_predictions: bool = True
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            save_predictions: Whether to save individual predictions
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_paths = []
        
        inference_times = []
        
        print("\nEvaluating model...")
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Evaluating"):
                # Handle different batch formats
                if len(batch_data) == 3:
                    images, targets, paths = batch_data
                else:
                    images, targets = batch_data
                    paths = None
                
                # Move to device
                images = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                inference_times.append(inference_time)
                
                # Collect predictions and targets
                predictions = outputs.cpu().numpy()
                targets = targets.numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                if paths is not None:
                    all_paths.extend(paths)
        
        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = compute_all_metrics(all_predictions, all_targets)
        
        # Add performance metrics
        metrics['avg_inference_time_ms'] = np.mean(inference_times)
        metrics['std_inference_time_ms'] = np.std(inference_times)
        metrics['total_samples'] = len(all_predictions)
        
        # Print results
        self.print_metrics(metrics)
        
        # Save results
        self.results = {
            'metrics': metrics,
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist()
        }
        
        if all_paths:
            self.results['paths'] = all_paths
        
        # Save to file
        self.save_results()
        
        if save_predictions:
            self.save_predictions(all_predictions, all_targets, all_paths)
        
        return metrics
    
    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics in a formatted way."""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"PLCC:  {metrics['plcc']:.4f} (p={metrics['plcc_pvalue']:.4e})")
        print(f"SRCC:  {metrics['srcc']:.4f} (p={metrics['srcc_pvalue']:.4e})")
        print(f"RMSE:  {metrics['rmse']:.4f}")
        print(f"MAE:   {metrics['mae']:.4f}")
        print(f"\nPerformance:")
        print(f"Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ± {metrics['std_inference_time_ms']:.2f} ms")
        print(f"Total Samples: {metrics['total_samples']}")
        print("="*60 + "\n")
    
    def save_results(self) -> None:
        """Save evaluation results to JSON file."""
        results_path = self.output_dir / 'evaluation_results.json'
        
        # Create a serializable version
        serializable_results = {
            'metrics': self.results['metrics'],
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Results saved to {results_path}")
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        paths: Optional[List[str]] = None
    ) -> None:
        """
        Save individual predictions to CSV file.
        
        Args:
            predictions: Predicted scores
            targets: Ground truth scores
            paths: Optional image paths
        """
        import pandas as pd
        
        data = {
            'prediction': predictions.flatten(),
            'target': targets.flatten(),
            'error': np.abs(predictions.flatten() - targets.flatten())
        }
        
        if paths:
            data['image_path'] = paths
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_dir / 'predictions.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"Predictions saved to {csv_path}")
    
    def cross_validate(
        self,
        data_loaders: List[DataLoader],
        fold_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform cross-validation evaluation.
        
        Args:
            data_loaders: List of data loaders for each fold
            fold_names: Optional names for each fold
        
        Returns:
            Dictionary with cross-validation results
        """
        n_folds = len(data_loaders)
        
        if fold_names is None:
            fold_names = [f"Fold {i+1}" for i in range(n_folds)]
        
        fold_results = []
        
        for fold_idx, (data_loader, fold_name) in enumerate(zip(data_loaders, fold_names)):
            print(f"\n{'='*60}")
            print(f"Evaluating {fold_name}")
            print(f"{'='*60}")
            
            metrics = self.evaluate(data_loader, save_predictions=False)
            fold_results.append(metrics)
        
        # Aggregate results
        metric_names = ['plcc', 'srcc', 'rmse', 'mae']
        aggregated = {}
        
        for metric_name in metric_names:
            values = [result[metric_name] for result in fold_results]
            mean, lower, upper = compute_confidence_interval(values)
            
            aggregated[metric_name] = {
                'mean': mean,
                'std': np.std(values),
                'ci_lower': lower,
                'ci_upper': upper,
                'values': values
            }
        
        # Print aggregated results
        print("\n" + "="*60)
        print("Cross-Validation Results (Mean ± Std)")
        print("="*60)
        for metric_name in metric_names:
            stats = aggregated[metric_name]
            print(f"{metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print("="*60 + "\n")
        
        # Save cross-validation results
        cv_results = {
            'fold_results': fold_results,
            'aggregated': {
                k: {kk: vv for kk, vv in v.items() if kk != 'values'}
                for k, v in aggregated.items()
            }
        }
        
        cv_path = self.output_dir / 'cross_validation_results.json'
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        print(f"Cross-validation results saved to {cv_path}")
        
        return aggregated
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader
    ) -> Dict:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model name -> model
            data_loader: Data loader for evaluation
        
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}")
            print(f"{'='*60}")
            
            # Temporarily switch model
            original_model = self.model
            self.model = model.to(self.device)
            
            metrics = self.evaluate(data_loader, save_predictions=False)
            comparison_results[model_name] = metrics
            
            # Restore original model
            self.model = original_model
        
        # Print comparison
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        print(f"{'Model':<20} {'PLCC':<10} {'SRCC':<10} {'RMSE':<10} {'MAE':<10}")
        print("-"*60)
        
        for model_name, metrics in comparison_results.items():
            print(f"{model_name:<20} {metrics['plcc']:<10.4f} "
                  f"{metrics['srcc']:<10.4f} {metrics['rmse']:<10.4f} "
                  f"{metrics['mae']:<10.4f}")
        
        print("="*60 + "\n")
        
        # Save comparison results
        comp_path = self.output_dir / 'model_comparison.json'
        with open(comp_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        print(f"Comparison results saved to {comp_path}")
        
        return comparison_results
