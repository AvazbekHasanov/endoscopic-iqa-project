"""
Image preprocessing utilities for endoscopic images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Image preprocessor for endoscopic images.
    Handles resizing, normalization, and basic preprocessing.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize using ImageNet statistics
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Preprocessed image
        """
        # Resize
        if image.shape[:2] != self.image_size:
            image = cv2.resize(
                image, (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization if requested
        if self.normalize:
            image = (image - self.mean) / self.std
        
        return image
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
        
        Returns:
            Batch of preprocessed images (B, H, W, 3)
        """
        processed = [self(img) for img in images]
        return np.stack(processed, axis=0)
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range.
        
        Args:
            image: Normalized image
        
        Returns:
            Denormalized image
        """
        if self.normalize:
            image = image * self.std + self.mean
        
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio, padding if necessary.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        pad_color: Color for padding
    
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # New dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Useful for enhancing endoscopic images with varying illumination.
    
    Args:
        image: Input RGB image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def remove_black_borders(
    image: np.ndarray,
    threshold: int = 10
) -> np.ndarray:
    """
    Remove black borders common in endoscopic images.
    
    Args:
        image: Input image
        threshold: Threshold for detecting black pixels
    
    Returns:
        Cropped image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find non-black pixels
    mask = gray > threshold
    
    # Find bounding box
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop image
    cropped = image[y_min:y_max + 1, x_min:x_max + 1]
    
    return cropped


def apply_color_correction(
    image: np.ndarray,
    method: str = 'gray_world'
) -> np.ndarray:
    """
    Apply color correction to endoscopic images.
    
    Args:
        image: Input RGB image
        method: Correction method ('gray_world' or 'white_patch')
    
    Returns:
        Color corrected image
    """
    image_float = image.astype(np.float32)
    
    if method == 'gray_world':
        # Gray world assumption: average color should be gray
        avg_r = np.mean(image_float[:, :, 0])
        avg_g = np.mean(image_float[:, :, 1])
        avg_b = np.mean(image_float[:, :, 2])
        
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        # Scale factors
        scale_r = avg_gray / (avg_r + 1e-6)
        scale_g = avg_gray / (avg_g + 1e-6)
        scale_b = avg_gray / (avg_b + 1e-6)
        
        # Apply correction
        corrected = image_float.copy()
        corrected[:, :, 0] *= scale_r
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_b
        
    elif method == 'white_patch':
        # White patch assumption: brightest pixel should be white
        max_r = np.max(image_float[:, :, 0])
        max_g = np.max(image_float[:, :, 1])
        max_b = np.max(image_float[:, :, 2])
        
        # Scale to make brightest pixel white
        corrected = image_float.copy()
        corrected[:, :, 0] *= 255.0 / (max_r + 1e-6)
        corrected[:, :, 1] *= 255.0 / (max_g + 1e-6)
        corrected[:, :, 2] *= 255.0 / (max_b + 1e-6)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected
