"""
Dataset loader for endoscopic images.
Supports multiple dataset formats and custom image loading.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path
import json


class EndoscopicDataset(Dataset):
    """
    Dataset class for endoscopic images with quality scores.
    
    Supports:
    - Loading images from directory structure
    - Custom quality score annotation files
    - On-the-fly preprocessing and augmentation
    - Synthetic degradation for training data
    """
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        degradation: Optional[Callable] = None,
        mode: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        return_path: bool = False
    ):
        """
        Initialize endoscopic image dataset.
        
        Args:
            data_dir: Root directory containing images
            annotation_file: Optional JSON file with image paths and quality scores
            transform: Optional transform/augmentation function
            degradation: Optional degradation function for synthetic data
            mode: Dataset mode ('train', 'val', 'test')
            image_size: Target image size (height, width)
            return_path: Whether to return image path along with data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.degradation = degradation
        self.mode = mode
        self.image_size = image_size
        self.return_path = return_path
        
        # Load image paths and scores
        self.samples = self._load_samples(annotation_file)
        
        print(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_samples(self, annotation_file: Optional[str]) -> List[Dict]:
        """Load image paths and quality scores."""
        samples = []
        
        if annotation_file and os.path.exists(annotation_file):
            # Load from annotation file
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for item in annotations:
                image_path = self.data_dir / item['image_path']
                if image_path.exists():
                    samples.append({
                        'path': str(image_path),
                        'score': item.get('quality_score', 0.0),
                        'mos': item.get('mos', 0.0)
                    })
        else:
            # Load all images from directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for ext in image_extensions:
                for image_path in self.data_dir.rglob(f'*{ext}'):
                    samples.append({
                        'path': str(image_path),
                        'score': 1.0,  # Default high quality
                        'mos': 5.0  # Default MOS score
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get image and quality score at index.
        
        Returns:
            If return_path=False: (image_tensor, quality_score)
            If return_path=True: (image_tensor, quality_score, image_path)
        """
        sample = self.samples[idx]
        image_path = sample['path']
        quality_score = sample['score']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply synthetic degradation if provided (training mode)
        if self.degradation and self.mode == 'train':
            image, quality_score = self.degradation(image, quality_score)
        
        # Resize image
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Apply transforms/augmentation
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        quality_score = torch.tensor(quality_score, dtype=torch.float32)
        
        if self.return_path:
            return image, quality_score, image_path
        return image, quality_score


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_annotation: Optional[str] = None,
    val_annotation: Optional[str] = None,
    test_annotation: Optional[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    degradation: Optional[Callable] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_annotation: Training annotation file
        val_annotation: Validation annotation file
        test_annotation: Test annotation file
        image_size: Target image size
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        degradation: Degradation function for training
        split_ratios: Train/val/test split ratios if annotations not provided
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create datasets
    train_dataset = EndoscopicDataset(
        data_dir=data_dir,
        annotation_file=train_annotation,
        transform=train_transform,
        degradation=degradation,
        mode='train',
        image_size=image_size
    )
    
    val_dataset = EndoscopicDataset(
        data_dir=data_dir,
        annotation_file=val_annotation,
        transform=val_transform,
        mode='val',
        image_size=image_size
    )
    
    test_dataset = EndoscopicDataset(
        data_dir=data_dir,
        annotation_file=test_annotation,
        transform=val_transform,
        mode='test',
        image_size=image_size,
        return_path=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class SimpleEndoscopicDataset(Dataset):
    """Simplified dataset for inference without annotations."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize simple dataset for inference.
        
        Args:
            image_paths: List of image file paths
            transform: Optional transform function
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get image tensor and path."""
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, image_path
