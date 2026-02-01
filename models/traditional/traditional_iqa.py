"""
Traditional (handcrafted) IQA metrics for endoscopic images.
Implements various no-reference image quality metrics.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import entropy
from typing import Dict, Optional, Tuple
import warnings


class TraditionalIQA:
    """
    Traditional IQA metrics implementation.
    
    Includes:
    - Laplacian focus measure (blur detection)
    - Gradient energy (sharpness)
    - RMS contrast
    - Entropy measure
    - Noise estimation
    - BRISQUE (simplified version)
    """
    
    def __init__(self):
        """Initialize traditional IQA metrics."""
        pass
    
    def compute_all_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute all traditional IQA metrics for an image.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
        
        Returns:
            Dictionary of metric names and values
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        metrics = {
            'laplacian_variance': self.laplacian_variance(gray),
            'gradient_energy': self.gradient_energy(gray),
            'rms_contrast': self.rms_contrast(gray),
            'entropy': self.image_entropy(gray),
            'noise_estimate': self.estimate_noise(gray),
            'tenengrad': self.tenengrad(gray),
            'mscn_std': self.mscn_std(gray)
        }
        
        return metrics
    
    def laplacian_variance(self, image: np.ndarray) -> float:
        """
        Compute Laplacian focus measure (blur detection).
        Formula: Q_blur = Var(∇²I)
        
        Higher values indicate sharper images.
        
        Args:
            image: Grayscale image
        
        Returns:
            Laplacian variance score
        """
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Compute variance
        variance = laplacian.var()
        
        return float(variance)
    
    def gradient_energy(self, image: np.ndarray) -> float:
        """
        Compute gradient energy (sharpness measure).
        Formula: Q_grad = Σ(Gx² + Gy²)
        
        Higher values indicate sharper images.
        
        Args:
            image: Grayscale image
        
        Returns:
            Gradient energy score
        """
        # Compute Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude squared
        grad_magnitude_sq = grad_x ** 2 + grad_y ** 2
        
        # Sum over all pixels
        energy = np.sum(grad_magnitude_sq)
        
        # Normalize by image size
        energy = energy / (image.shape[0] * image.shape[1])
        
        return float(energy)
    
    def rms_contrast(self, image: np.ndarray) -> float:
        """
        Compute RMS (Root Mean Square) contrast.
        Formula: Q_contrast = √(1/N * Σ(Ii - μ)²)
        
        Higher values indicate higher contrast.
        
        Args:
            image: Grayscale image (0-255 range)
        
        Returns:
            RMS contrast score
        """
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float64) / 255.0
        
        # Compute mean intensity
        mean_intensity = np.mean(image_normalized)
        
        # Compute RMS contrast
        contrast = np.sqrt(np.mean((image_normalized - mean_intensity) ** 2))
        
        return float(contrast)
    
    def image_entropy(self, image: np.ndarray) -> float:
        """
        Compute Shannon entropy of image intensity distribution.
        Formula: Q_entropy = -Σ pi * log₂(pi)
        
        Higher values indicate more information content.
        
        Args:
            image: Grayscale image (0-255 range)
        
        Returns:
            Entropy score
        """
        # Compute histogram
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        
        # Normalize to get probabilities
        hist = hist.astype(np.float64)
        hist = hist / hist.sum()
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Compute entropy
        ent = -np.sum(hist * np.log2(hist))
        
        return float(ent)
    
    def estimate_noise(self, image: np.ndarray, window_size: int = 7) -> float:
        """
        Estimate noise level using local variance method.
        Formula: Q_noise = 1/M * Σ Var(Rk)
        
        Lower values indicate less noise.
        
        Args:
            image: Grayscale image
            window_size: Size of local window
        
        Returns:
            Noise estimate
        """
        # Convert to float
        image_float = image.astype(np.float64)
        
        # Compute local mean using uniform filter
        local_mean = ndimage.uniform_filter(image_float, size=window_size)
        
        # Compute local variance
        local_var = ndimage.uniform_filter(
            image_float ** 2, size=window_size
        ) - local_mean ** 2
        
        # Estimate noise as mean of local standard deviations
        # (in homogeneous regions)
        # Use median to be robust to edges
        noise_estimate = np.median(np.sqrt(np.abs(local_var)))
        
        return float(noise_estimate)
    
    def tenengrad(self, image: np.ndarray, ksize: int = 3) -> float:
        """
        Compute Tenengrad focus measure (alternative sharpness metric).
        Based on Sobel gradient magnitude.
        
        Args:
            image: Grayscale image
            ksize: Sobel kernel size
        
        Returns:
            Tenengrad score
        """
        # Compute Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Threshold to focus on edges
        threshold = grad_magnitude.mean()
        grad_magnitude[grad_magnitude < threshold] = 0
        
        # Sum of squared gradients
        tenengrad_score = np.sum(grad_magnitude ** 2)
        
        # Normalize by image size
        tenengrad_score = tenengrad_score / (image.shape[0] * image.shape[1])
        
        return float(tenengrad_score)
    
    def mscn_coefficients(
        self, image: np.ndarray, sigma: float = 7.0, C: float = 1.0
    ) -> np.ndarray:
        """
        Compute Mean Subtracted Contrast Normalized (MSCN) coefficients.
        Used in BRISQUE algorithm.
        
        Formula: Î(x,y) = (I(x,y) - μ(x,y)) / (σ(x,y) + C)
        
        Args:
            image: Grayscale image (0-255 range)
            sigma: Gaussian kernel standard deviation
            C: Constant for numerical stability
        
        Returns:
            MSCN coefficients
        """
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float64) / 255.0
        
        # Compute local mean
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        mu = cv2.GaussianBlur(image_normalized, (kernel_size, kernel_size), sigma)
        
        # Compute local variance
        mu_sq = cv2.GaussianBlur(
            image_normalized ** 2, (kernel_size, kernel_size), sigma
        )
        sigma_map = np.sqrt(np.abs(mu_sq - mu ** 2))
        
        # Compute MSCN coefficients
        mscn = (image_normalized - mu) / (sigma_map + C / 255.0)
        
        return mscn
    
    def mscn_std(self, image: np.ndarray) -> float:
        """
        Compute standard deviation of MSCN coefficients.
        Part of BRISQUE feature extraction.
        
        Args:
            image: Grayscale image
        
        Returns:
            Standard deviation of MSCN coefficients
        """
        mscn = self.mscn_coefficients(image)
        return float(np.std(mscn))
    
    def compute_quality_score(
        self, image: np.ndarray, method: str = 'combined'
    ) -> float:
        """
        Compute overall quality score using traditional metrics.
        
        Args:
            image: Input image
            method: Method for combining metrics
                   - 'combined': Weighted combination of all metrics
                   - 'laplacian': Only Laplacian variance
                   - 'gradient': Only gradient energy
        
        Returns:
            Quality score (0-1, higher is better)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == 'laplacian':
            # Normalize Laplacian variance to [0, 1]
            lap_var = self.laplacian_variance(gray)
            # Typical range is 0-1000 for good images
            score = min(lap_var / 1000.0, 1.0)
            
        elif method == 'gradient':
            # Normalize gradient energy
            grad_energy = self.gradient_energy(gray)
            # Typical range is 0-10000
            score = min(grad_energy / 10000.0, 1.0)
            
        elif method == 'combined':
            # Compute all metrics
            metrics = self.compute_all_metrics(image)
            
            # Normalize metrics to [0, 1] range
            lap_norm = min(metrics['laplacian_variance'] / 1000.0, 1.0)
            grad_norm = min(metrics['gradient_energy'] / 10000.0, 1.0)
            contrast_norm = min(metrics['rms_contrast'] * 5.0, 1.0)
            entropy_norm = metrics['entropy'] / 8.0  # Max entropy ~8 for 256 levels
            noise_norm = max(1.0 - metrics['noise_estimate'] / 50.0, 0.0)
            
            # Weighted combination
            weights = {
                'laplacian': 0.3,
                'gradient': 0.25,
                'contrast': 0.2,
                'entropy': 0.15,
                'noise': 0.1
            }
            
            score = (
                weights['laplacian'] * lap_norm +
                weights['gradient'] * grad_norm +
                weights['contrast'] * contrast_norm +
                weights['entropy'] * entropy_norm +
                weights['noise'] * noise_norm
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(score)
    
    def assess_blur(self, image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
        """
        Assess if image is blurry.
        
        Args:
            image: Input image
            threshold: Threshold for blur detection (lower = more blur)
        
        Returns:
            Tuple of (is_blurry, blur_score)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        blur_score = self.laplacian_variance(gray)
        is_blurry = blur_score < threshold
        
        return is_blurry, blur_score
    
    def assess_noise(self, image: np.ndarray, threshold: float = 15.0) -> Tuple[bool, float]:
        """
        Assess if image is noisy.
        
        Args:
            image: Input image
            threshold: Threshold for noise detection (higher = more noise)
        
        Returns:
            Tuple of (is_noisy, noise_level)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        noise_level = self.estimate_noise(gray)
        is_noisy = noise_level > threshold
        
        return is_noisy, noise_level
