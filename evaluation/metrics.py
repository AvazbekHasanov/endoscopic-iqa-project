"""
Evaluation metrics for Image Quality Assessment.
Implements correlation metrics and error metrics.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_plcc(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Pearson Linear Correlation Coefficient (PLCC).
    Measures linear correlation between predictions and ground truth.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
    
    Returns:
        Tuple of (PLCC value, p-value)
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) < 2:
        return 0.0, 1.0
    
    plcc, p_value = pearsonr(predictions, targets)
    return float(plcc), float(p_value)


def compute_srcc(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Spearman Rank Correlation Coefficient (SRCC).
    Measures monotonic correlation between predictions and ground truth.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
    
    Returns:
        Tuple of (SRCC value, p-value)
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) < 2:
        return 0.0, 1.0
    
    srcc, p_value = spearmanr(predictions, targets)
    return float(srcc), float(p_value)


def compute_rmse(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Root Mean Square Error (RMSE).
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
    
    Returns:
        RMSE value
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return float('inf')
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    return float(rmse)


def compute_mae(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
    
    Returns:
        MAE value
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return float('inf')
    
    mae = mean_absolute_error(targets, predictions)
    return float(mae)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth scores
    
    Returns:
        Dictionary containing all metrics
    """
    plcc, plcc_p = compute_plcc(predictions, targets)
    srcc, srcc_p = compute_srcc(predictions, targets)
    rmse = compute_rmse(predictions, targets)
    mae = compute_mae(predictions, targets)
    
    metrics = {
        'plcc': plcc,
        'plcc_pvalue': plcc_p,
        'srcc': srcc,
        'srcc_pvalue': srcc_p,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: List of metric values (e.g., from cross-validation)
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # For small samples, use t-distribution
    if len(values) < 30:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        margin = t_value * std / np.sqrt(len(values))
    else:
        # For large samples, use normal distribution
        z_value = 1.96  # For 95% confidence
        margin = z_value * std / np.sqrt(len(values))
    
    lower = mean - margin
    upper = mean + margin
    
    return float(mean), float(lower), float(upper)


def compute_correlation_matrix(
    predictions_list: List[np.ndarray],
    targets: np.ndarray,
    method_names: List[str]
) -> np.ndarray:
    """
    Compute correlation matrix between different methods.
    
    Args:
        predictions_list: List of prediction arrays from different methods
        targets: Ground truth scores
        method_names: Names of methods
    
    Returns:
        Correlation matrix
    """
    n_methods = len(predictions_list)
    corr_matrix = np.zeros((n_methods, n_methods))
    
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(
                    predictions_list[i].flatten(),
                    predictions_list[j].flatten()
                )
                corr_matrix[i, j] = corr
    
    return corr_matrix


def statistical_significance_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    targets: np.ndarray,
    test: str = 'wilcoxon'
) -> Tuple[float, float]:
    """
    Test statistical significance between two methods.
    
    Args:
        predictions1: Predictions from method 1
        predictions2: Predictions from method 2
        targets: Ground truth scores
        test: Statistical test to use ('wilcoxon' or 't-test')
    
    Returns:
        Tuple of (test statistic, p-value)
    """
    from scipy import stats
    
    # Compute errors
    errors1 = np.abs(predictions1.flatten() - targets.flatten())
    errors2 = np.abs(predictions2.flatten() - targets.flatten())
    
    if test == 'wilcoxon':
        # Wilcoxon signed-rank test (non-parametric)
        statistic, p_value = stats.wilcoxon(errors1, errors2)
    elif test == 't-test':
        # Paired t-test (parametric)
        statistic, p_value = stats.ttest_rel(errors1, errors2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return float(statistic), float(p_value)
