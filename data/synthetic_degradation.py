"""
Synthetic degradation pipeline for endoscopic images.
Implements various degradation types common in endoscopy.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Callable, Dict
import random


class SyntheticDegradation:
    """
    Synthetic degradation pipeline for endoscopic images.
    
    Supports multiple degradation types:
    - Motion blur
    - Defocus blur
    - Gaussian noise
    - Poisson noise
    - Illumination variations
    - Specular reflections
    - Color distortions
    """
    
    def __init__(
        self,
        degradation_types: Optional[list] = None,
        severity_range: Tuple[float, float] = (0.1, 0.9)
    ):
        """
        Initialize degradation pipeline.
        
        Args:
            degradation_types: List of degradation types to use. If None, uses all.
            severity_range: Range for degradation severity (min, max)
        """
        self.severity_range = severity_range
        
        # Available degradation functions
        self.degradation_funcs = {
            'motion_blur': self.apply_motion_blur,
            'defocus_blur': self.apply_defocus_blur,
            'gaussian_noise': self.apply_gaussian_noise,
            'poisson_noise': self.apply_poisson_noise,
            'illumination': self.apply_illumination_variation,
            'specular': self.apply_specular_reflection,
            'color_distortion': self.apply_color_distortion
        }
        
        if degradation_types is None:
            self.degradation_types = list(self.degradation_funcs.keys())
        else:
            self.degradation_types = degradation_types
    
    def apply_random_degradation(
        self,
        image: np.ndarray,
        base_quality: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply random degradation to image and compute quality score.
        
        Args:
            image: Input image (H, W, 3)
            base_quality: Base quality score before degradation
        
        Returns:
            Tuple of (degraded_image, quality_score)
        """
        # Randomly select degradation type
        deg_type = random.choice(self.degradation_types)
        
        # Random severity level
        severity = random.uniform(*self.severity_range)
        
        # Apply degradation
        degraded_image = self.degradation_funcs[deg_type](image, severity)
        
        # Compute quality score based on severity
        # Higher severity = lower quality
        quality_score = base_quality * (1.0 - severity * 0.8)  # Quality drops up to 80%
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        return degraded_image, quality_score
    
    def apply_motion_blur(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply motion blur to simulate camera or tissue movement.
        
        Args:
            image: Input image
            severity: Blur severity (0-1)
        
        Returns:
            Motion blurred image
        """
        # Kernel size based on severity
        kernel_size = int(3 + severity * 20)  # 3 to 23 pixels
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Random motion direction
        angle = random.uniform(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D(
            (kernel_size // 2, kernel_size // 2), angle, 1.0
        )
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def apply_defocus_blur(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply defocus blur (out-of-focus effect).
        
        Args:
            image: Input image
            severity: Blur severity (0-1)
        
        Returns:
            Defocused image
        """
        # Kernel size based on severity
        kernel_size = int(3 + severity * 20)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def apply_gaussian_noise(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add Gaussian noise to simulate sensor noise.
        
        Args:
            image: Input image
            severity: Noise severity (0-1)
        
        Returns:
            Noisy image
        """
        # Noise standard deviation based on severity
        std = severity * 50.0  # 0 to 50 std
        
        # Generate Gaussian noise
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        
        # Add noise
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def apply_poisson_noise(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add Poisson noise to simulate low-light conditions.
        
        Args:
            image: Input image
            severity: Noise severity (0-1)
        
        Returns:
            Noisy image
        """
        # Scale factor based on severity (lower = more noise)
        scale = 1.0 - severity * 0.9  # Keep at least 10%
        
        # Scale image
        scaled = image.astype(np.float32) * scale
        
        # Apply Poisson noise
        noisy = np.random.poisson(scaled).astype(np.float32)
        
        # Scale back
        noisy = noisy / scale
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def apply_illumination_variation(
        self, image: np.ndarray, severity: float
    ) -> np.ndarray:
        """
        Apply uneven illumination to simulate lighting variations.
        
        Args:
            image: Input image
            severity: Variation severity (0-1)
        
        Returns:
            Image with illumination variation
        """
        h, w = image.shape[:2]
        
        # Create illumination map
        # Random center point
        cx = random.uniform(0.3 * w, 0.7 * w)
        cy = random.uniform(0.3 * h, 0.7 * h)
        
        # Create meshgrid
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2) / 2
        
        # Normalized distance
        norm_dist = dist / max_dist
        
        # Illumination falloff
        illumination = 1.0 - severity * norm_dist
        illumination = np.clip(illumination, 0.2, 1.0)
        
        # Apply to each channel
        illuminated = image.astype(np.float32) * illumination[:, :, np.newaxis]
        illuminated = np.clip(illuminated, 0, 255).astype(np.uint8)
        
        return illuminated
    
    def apply_specular_reflection(
        self, image: np.ndarray, severity: float
    ) -> np.ndarray:
        """
        Add specular reflections (bright spots).
        
        Args:
            image: Input image
            severity: Reflection severity (0-1)
        
        Returns:
            Image with specular reflections
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # Number of reflections based on severity
        num_reflections = int(1 + severity * 5)  # 1 to 6 reflections
        
        for _ in range(num_reflections):
            # Random position
            cx = random.randint(int(0.2 * w), int(0.8 * w))
            cy = random.randint(int(0.2 * h), int(0.8 * h))
            
            # Random size based on severity
            radius = int(10 + severity * 50)  # 10 to 60 pixels
            
            # Create reflection mask
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            
            # Gaussian falloff
            reflection = np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2))
            reflection = reflection[:, :, np.newaxis]
            
            # Add bright spot
            intensity = severity * 200  # Brightness
            result = result + reflection * intensity
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def apply_color_distortion(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply color distortion (white balance and color shifts).
        
        Args:
            image: Input image
            severity: Distortion severity (0-1)
        
        Returns:
            Color distorted image
        """
        # Random color shift per channel
        shifts = [
            random.uniform(-severity * 50, severity * 50) for _ in range(3)
        ]
        
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] += shifts[i]
        
        # Random contrast change
        contrast = 1.0 + random.uniform(-severity * 0.5, severity * 0.5)
        mean = result.mean()
        result = (result - mean) * contrast + mean
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def compute_mos_score(self, severity: float, deg_type: str) -> float:
        """
        Compute Mean Opinion Score based on degradation.
        
        Args:
            severity: Degradation severity (0-1)
            deg_type: Degradation type
        
        Returns:
            MOS score (1-5 scale)
        """
        # Base MOS is 5 (excellent)
        base_mos = 5.0
        
        # Different degradation types have different impact
        impact_weights = {
            'motion_blur': 1.2,
            'defocus_blur': 1.0,
            'gaussian_noise': 0.9,
            'poisson_noise': 1.1,
            'illumination': 0.7,
            'specular': 1.3,
            'color_distortion': 0.6
        }
        
        weight = impact_weights.get(deg_type, 1.0)
        
        # MOS decreases with severity
        mos = base_mos - severity * 4.0 * weight
        mos = np.clip(mos, 1.0, 5.0)
        
        return mos
    
    def apply_multiple_degradations(
        self,
        image: np.ndarray,
        num_degradations: int = 2,
        base_quality: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply multiple random degradations sequentially.
        
        Args:
            image: Input image
            num_degradations: Number of degradations to apply
            base_quality: Base quality score
        
        Returns:
            Tuple of (degraded_image, quality_score)
        """
        result = image.copy()
        cumulative_severity = 0.0
        
        # Randomly select degradation types
        selected_types = random.sample(
            self.degradation_types,
            min(num_degradations, len(self.degradation_types))
        )
        
        for deg_type in selected_types:
            severity = random.uniform(*self.severity_range)
            result = self.degradation_funcs[deg_type](result, severity)
            cumulative_severity += severity
        
        # Average severity
        avg_severity = cumulative_severity / num_degradations
        
        # Compute quality score
        quality_score = base_quality * (1.0 - avg_severity * 0.8)
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        return result, quality_score
