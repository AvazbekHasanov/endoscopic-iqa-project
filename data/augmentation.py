"""
Data augmentation utilities for endoscopic image quality assessment.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_augmentation_pipeline(
    mode: str = 'train',
    image_size: tuple = (224, 224),
    normalize: bool = True
) -> A.Compose:
    """
    Get augmentation pipeline for training or validation.
    
    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size (height, width)
        normalize: Whether to normalize using ImageNet statistics
    
    Returns:
        Albumentations compose pipeline
    """
    if mode == 'train':
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            # Geometric transforms (careful with medical images)
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            # Blur and noise (light application)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        ]
    else:
        # Validation/test - only resize
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
        ]
    
    # Add normalization if requested
    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        )
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_light_augmentation(
    image_size: tuple = (224, 224)
) -> A.Compose:
    """
    Light augmentation pipeline suitable for medical images.
    
    Args:
        image_size: Target image size
    
    Returns:
        Light augmentation pipeline
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_test_augmentation(
    image_size: tuple = (224, 224),
    normalize: bool = True
) -> A.Compose:
    """
    Test-time augmentation pipeline (TTA).
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize
    
    Returns:
        TTA pipeline
    """
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]
    
    if normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_endoscopic_augmentation(
    image_size: tuple = (224, 224),
    aggressive: bool = False
) -> A.Compose:
    """
    Specialized augmentation for endoscopic images.
    
    Args:
        image_size: Target image size
        aggressive: Whether to use more aggressive augmentation
    
    Returns:
        Endoscopic-specific augmentation pipeline
    """
    if aggressive:
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=30,
                p=0.6
            ),
            # Color variations common in endoscopy
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=15,
                p=0.5
            ),
            # Simulate lighting variations
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # Blur effects
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            # Compression artifacts
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        ]
    else:
        transforms = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
        ]
    
    # Always add normalization and tensor conversion
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)
