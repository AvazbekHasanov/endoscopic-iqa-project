"""
Hybrid IQA Predictor combining Traditional and Deep Learning approaches.
Provides both individual and ensemble predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.traditional.traditional_iqa import TraditionalIQA
from models.deep_learning.iqa_model import get_model


class HybridIQAPredictor:
    """
    Hybrid IQA predictor combining traditional and deep learning methods.

    Provides:
    - Traditional metrics-based quality score
    - Deep learning CNN-based quality score
    - Ensemble score (weighted combination)
    """

    def __init__(
        self,
        dl_model_path: Optional[str] = None,
        model_type: str = 'lightweight',
        device: str = 'auto',
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hybrid predictor.

        Args:
            dl_model_path: Path to pretrained deep learning model (optional)
            model_type: Type of DL model ('lightweight' or 'full')
            device: Device to run DL model on ('cpu', 'cuda', or 'auto')
            ensemble_weights: Weights for ensemble {traditional: x, deep_learning: y}
        """
        # Initialize traditional IQA
        self.traditional_iqa = TraditionalIQA()

        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize deep learning model
        self.dl_model = get_model(model_type=model_type)

        # Auto-detect pretrained model if not specified
        if dl_model_path is None:
            # Check standard locations
            pretrained_dir = Path(__file__).parent / 'pretrained'
            possible_models = [
                pretrained_dir / 'mobilenet_v2_iqa.pth',
                pretrained_dir / 'resnet18_iqa.pth',
                pretrained_dir / 'best_model.pth',
                pretrained_dir / 'trained_model.pth'
            ]

            for model_path in possible_models:
                if model_path.exists():
                    dl_model_path = str(model_path)
                    print(f"🔍 Found pretrained model: {model_path.name}")
                    break

        # Load pretrained weights if provided or found
        if dl_model_path and Path(dl_model_path).exists():
            self.load_model(dl_model_path)
            self.dl_model_loaded = True
        else:
            print(f"⚠️  No pretrained model found. Deep learning predictions will use untrained model.")
            print(f"   Run: python3 scripts/setup_pretrained_model.py to download one!")
            print(f"   Traditional IQA will still work perfectly!")
            self.dl_model_loaded = False

        self.dl_model = self.dl_model.to(self.device)
        self.dl_model.eval()

        # Set ensemble weights
        if ensemble_weights is None:
            # Default: Equal weights if both models available, otherwise 100% traditional
            if self.dl_model_loaded:
                self.ensemble_weights = {'traditional': 0.5, 'deep_learning': 0.5}
            else:
                self.ensemble_weights = {'traditional': 1.0, 'deep_learning': 0.0}
        else:
            self.ensemble_weights = ensemble_weights

        # Normalization for deep learning input (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def load_model(self, model_path: str):
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.dl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.dl_model.load_state_dict(checkpoint)
            print(f"✓ Loaded deep learning model from: {model_path}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print(f"   Using untrained model. Traditional IQA will still work!")

    def preprocess_image_traditional(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for traditional IQA.

        Args:
            image: Input image (H, W, 3) RGB or BGR

        Returns:
            Preprocessed image
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def preprocess_image_dl(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Preprocess image for deep learning model.

        Args:
            image: Input image (H, W, 3) RGB format
            target_size: Target size for model input

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Resize
        image_resized = cv2.resize(image, target_size)

        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Normalize using ImageNet stats
        image_tensor = image_tensor.to(self.device)
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def predict_traditional(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict quality using traditional methods.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Dictionary with traditional metrics and score
        """
        image = self.preprocess_image_traditional(image)

        # Compute all traditional metrics
        metrics = self.traditional_iqa.compute_all_metrics(image)

        # Compute overall quality score
        quality_score = self.traditional_iqa.compute_quality_score(image, method='combined')

        return {
            'quality_score': quality_score,
            'metrics': metrics
        }

    def predict_deep_learning(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> float:
        """
        Predict quality using deep learning model.

        Args:
            image: Input image (H, W, 3)
            target_size: Target size for model

        Returns:
            Quality score (0-1)
        """
        if not self.dl_model_loaded:
            print("⚠️  Deep learning model not loaded, returning 0.5 as placeholder")
            return 0.5

        image_tensor = self.preprocess_image_dl(image, target_size)

        with torch.no_grad():
            score = self.dl_model(image_tensor)
            score = score.item()

        return score

    def predict_ensemble(
        self,
        image: np.ndarray,
        return_individual: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Predict quality using ensemble of traditional and deep learning.

        Args:
            image: Input image (H, W, 3)
            return_individual: Whether to return individual scores

        Returns:
            Ensemble quality score or dict with all scores
        """
        # Get traditional prediction
        trad_result = self.predict_traditional(image)
        trad_score = trad_result['quality_score']

        # Get deep learning prediction
        dl_score = self.predict_deep_learning(image)

        # Compute ensemble score
        ensemble_score = (
            self.ensemble_weights['traditional'] * trad_score +
            self.ensemble_weights['deep_learning'] * dl_score
        )

        if return_individual:
            return {
                'ensemble_score': ensemble_score,
                'traditional_score': trad_score,
                'deep_learning_score': dl_score,
                'traditional_metrics': trad_result['metrics']
            }
        else:
            return ensemble_score

    def predict(
        self,
        image: Union[str, np.ndarray],
        method: str = 'ensemble',
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Main prediction method.

        Args:
            image: Input image path or numpy array
            method: Prediction method ('traditional', 'deep_learning', 'ensemble')
            return_details: Whether to return detailed results

        Returns:
            Quality score or detailed results dict
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict based on method
        if method == 'traditional':
            result = self.predict_traditional(image)
            if return_details:
                return result
            else:
                return result['quality_score']

        elif method == 'deep_learning':
            score = self.predict_deep_learning(image)
            if return_details:
                return {'quality_score': score}
            else:
                return score

        elif method == 'ensemble':
            return self.predict_ensemble(image, return_individual=return_details)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'traditional', 'deep_learning', or 'ensemble'")

    def batch_predict(
        self,
        images: list,
        method: str = 'ensemble',
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict quality for a batch of images.

        Args:
            images: List of image paths or numpy arrays
            method: Prediction method
            batch_size: Batch size for deep learning inference

        Returns:
            Array of quality scores
        """
        scores = []

        for image in images:
            score = self.predict(image, method=method, return_details=False)
            scores.append(score)

        return np.array(scores)

    def set_ensemble_weights(self, traditional: float, deep_learning: float):
        """
        Set custom ensemble weights.

        Args:
            traditional: Weight for traditional methods
            deep_learning: Weight for deep learning
        """
        total = traditional + deep_learning
        self.ensemble_weights = {
            'traditional': traditional / total,
            'deep_learning': deep_learning / total
        }
        print(f"✓ Ensemble weights updated: Traditional={self.ensemble_weights['traditional']:.2f}, "
              f"Deep Learning={self.ensemble_weights['deep_learning']:.2f}")

    def get_info(self) -> Dict:
        """Get predictor information."""
        return {
            'device': str(self.device),
            'dl_model_loaded': self.dl_model_loaded,
            'ensemble_weights': self.ensemble_weights,
            'cuda_available': torch.cuda.is_available()
        }


def main():
    """Demo usage of hybrid predictor."""
    import time

    print("=" * 80)
    print("🔬 HYBRID IQA PREDICTOR DEMO")
    print("=" * 80)

    # Initialize predictor
    print("\n📦 Initializing hybrid predictor...")
    predictor = HybridIQAPredictor(
        dl_model_path=None,  # No pretrained model yet
        model_type='lightweight',
        device='auto'
    )

    info = predictor.get_info()
    print(f"✓ Device: {info['device']}")
    print(f"✓ DL Model Loaded: {info['dl_model_loaded']}")
    print(f"✓ Ensemble Weights: {info['ensemble_weights']}")

    # Load sample image
    print("\n📸 Testing with sample image...")

    # Create a dummy image for demo
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Test traditional method
    print("\n1️⃣  Traditional IQA:")
    start = time.time()
    trad_result = predictor.predict(dummy_image, method='traditional', return_details=True)
    trad_time = (time.time() - start) * 1000
    print(f"   Score: {trad_result['quality_score']:.4f}")
    print(f"   Time: {trad_time:.2f}ms")
    print(f"   Metrics: {list(trad_result['metrics'].keys())}")

    # Test deep learning method
    print("\n2️⃣  Deep Learning IQA:")
    start = time.time()
    dl_score = predictor.predict(dummy_image, method='deep_learning')
    dl_time = (time.time() - start) * 1000
    print(f"   Score: {dl_score:.4f}")
    print(f"   Time: {dl_time:.2f}ms")

    # Test ensemble method
    print("\n3️⃣  Ensemble IQA:")
    start = time.time()
    ensemble_result = predictor.predict(dummy_image, method='ensemble', return_details=True)
    ensemble_time = (time.time() - start) * 1000
    print(f"   Ensemble Score: {ensemble_result['ensemble_score']:.4f}")
    print(f"   Traditional: {ensemble_result['traditional_score']:.4f}")
    print(f"   Deep Learning: {ensemble_result['deep_learning_score']:.4f}")
    print(f"   Time: {ensemble_time:.2f}ms")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

