"""
Basic usage examples for the Endoscopic IQA system.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.traditional.traditional_iqa import TraditionalIQA
from models.deep_learning import get_model
from inference.predictor import IQAPredictor


def example_traditional_metrics():
    """Example: Using traditional IQA metrics."""
    print("="*60)
    print("Example 1: Traditional IQA Metrics")
    print("="*60)
    
    # Create a sample image (or load from file)
    # For demo, we'll create a synthetic image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize traditional IQA
    iqa = TraditionalIQA()
    
    # Compute all metrics
    metrics = iqa.compute_all_metrics(image)
    
    print("\nTraditional IQA Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Compute overall quality score
    quality_score = iqa.compute_quality_score(image, method='combined')
    print(f"\nOverall Quality Score: {quality_score:.4f}")
    
    # Check for blur
    is_blurry, blur_score = iqa.assess_blur(image)
    print(f"Is Blurry: {is_blurry}, Blur Score: {blur_score:.4f}")
    
    # Check for noise
    is_noisy, noise_level = iqa.assess_noise(image)
    print(f"Is Noisy: {is_noisy}, Noise Level: {noise_level:.4f}")
    
    print()


def example_deep_learning_model():
    """Example: Using deep learning model."""
    print("="*60)
    print("Example 2: Deep Learning IQA Model")
    print("="*60)
    
    # Create model
    model = get_model(model_type='lightweight', base_channels=32)
    
    print(f"\nModel created successfully!")
    print(f"Model type: Lightweight CNN")
    
    # Count parameters
    from models.deep_learning.iqa_model import count_parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")
    
    # Get model size
    from models.utils import get_model_size
    model_size = get_model_size(model)
    print(f"Model size: {model_size:.2f} MB")
    
    print()


def example_predictor():
    """Example: Using predictor for inference."""
    print("="*60)
    print("Example 3: Using IQA Predictor")
    print("="*60)
    
    # Create a sample image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create model and predictor
    model = get_model(model_type='lightweight')
    predictor = IQAPredictor(model=model, device='cpu')
    
    print("\nPredictor initialized successfully!")
    
    # Predict quality score
    score, inference_time = predictor.predict(image, return_time=True)
    
    print(f"Quality Score: {score:.4f}")
    print(f"Inference Time: {inference_time:.2f} ms")
    
    # Get quality category
    category = predictor.get_quality_category(score)
    print(f"Quality Category: {category}")
    
    print()


def example_synthetic_degradation():
    """Example: Using synthetic degradation."""
    print("="*60)
    print("Example 4: Synthetic Degradation")
    print("="*60)
    
    from data.synthetic_degradation import SyntheticDegradation
    
    # Create clean image
    image = np.ones((256, 256, 3), dtype=np.uint8) * 200  # Light gray
    
    # Initialize degradation pipeline
    degrader = SyntheticDegradation()
    
    print("\nApplying different degradations:")
    
    # Apply motion blur
    degraded, quality = degrader.apply_motion_blur(image, severity=0.5)
    print(f"Motion Blur - Quality: {quality:.4f}")
    
    # Apply Gaussian noise
    degraded, quality = degrader.apply_gaussian_noise(image, severity=0.3)
    print(f"Gaussian Noise - Quality: {quality:.4f}")
    
    # Apply random degradation
    degraded, quality = degrader.apply_random_degradation(image)
    print(f"Random Degradation - Quality: {quality:.4f}")
    
    # Apply multiple degradations
    degraded, quality = degrader.apply_multiple_degradations(image, num_degradations=2)
    print(f"Multiple Degradations - Quality: {quality:.4f}")
    
    print()


def example_data_loading():
    """Example: Loading and preprocessing data."""
    print("="*60)
    print("Example 5: Data Loading and Preprocessing")
    print("="*60)
    
    from data.preprocessing import ImagePreprocessor
    from data.augmentation import get_augmentation_pipeline
    
    # Create sample image
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(image_size=(224, 224), normalize=True)
    
    print("\nPreprocessing image...")
    processed = preprocessor(image)
    print(f"Original shape: {image.shape}")
    print(f"Processed shape: {processed.shape}")
    
    # Get augmentation pipeline
    train_aug = get_augmentation_pipeline(mode='train', image_size=(224, 224))
    print("\nTrain augmentation pipeline created")
    
    val_aug = get_augmentation_pipeline(mode='val', image_size=(224, 224))
    print("Validation augmentation pipeline created")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Endoscopic IQA - Basic Usage Examples")
    print("="*60 + "\n")
    
    # Run examples
    example_traditional_metrics()
    example_deep_learning_model()
    example_predictor()
    example_synthetic_degradation()
    example_data_loading()
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
