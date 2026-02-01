"""
Predictor class for IQA inference.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
import time

from data.preprocessing import ImagePreprocessor
from models.utils import load_checkpoint


class IQAPredictor:
    """
    Predictor for image quality assessment.
    Handles single image and batch predictions with efficient preprocessing.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        image_size: tuple = (224, 224),
        batch_size: int = 32
    ):
        """
        Initialize predictor.
        
        Args:
            model: Pre-initialized model (if not provided, loads from model_path)
            model_path: Path to model checkpoint
            device: Device for inference
            image_size: Input image size
            batch_size: Batch size for batch prediction
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
            normalize=True
        )
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Predictor initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        from models.deep_learning import get_model
        
        # Create model (you might need to adjust model type)
        model = get_model(model_type='lightweight')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        return_time: bool = False
    ) -> Union[float, tuple]:
        """
        Predict quality score for a single image.
        
        Args:
            image: Image path or numpy array
            return_time: Whether to return inference time
        
        Returns:
            Quality score (and optionally inference time in ms)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        processed = self.preprocessor(image)
        
        # Convert to tensor and add batch dimension
        if not isinstance(processed, torch.Tensor):
            processed = torch.from_numpy(processed).permute(2, 0, 1)
        
        processed = processed.unsqueeze(0).to(self.device)
        
        # Predict
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(processed)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        score = output.item()
        
        if return_time:
            return score, inference_time
        return score
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Predict quality scores for a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            show_progress: Whether to show progress bar
        
        Returns:
            Array of quality scores
        """
        scores = []
        
        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Predicting")
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Load and preprocess batch
            processed_batch = []
            for img in batch:
                if isinstance(img, str):
                    img = cv2.imread(img)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                processed = self.preprocessor(img)
                if not isinstance(processed, torch.Tensor):
                    processed = torch.from_numpy(processed).permute(2, 0, 1)
                processed_batch.append(processed)
            
            if len(processed_batch) == 0:
                continue
            
            # Stack and predict
            batch_tensor = torch.stack(processed_batch).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            scores.extend(outputs.cpu().numpy().flatten())
        
        return np.array(scores)
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 1,
        show_preview: bool = False
    ) -> List[float]:
        """
        Predict quality scores for video frames.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            sample_rate: Sample every N frames
            show_preview: Whether to show preview window
        
        Returns:
            List of quality scores for sampled frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        scores = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                # Convert to RGB for prediction
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict
                score = self.predict(frame_rgb)
                scores.append(score)
                
                # Annotate frame
                if output_path or show_preview:
                    cv2.putText(
                        frame,
                        f"Quality: {score:.3f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.4 else (0, 0, 255),
                        2
                    )
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Video Quality Assessment', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"Processed {len(scores)} frames")
        if output_path:
            print(f"Annotated video saved to {output_path}")
        
        return scores
    
    def predict_directory(
        self,
        directory: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        recursive: bool = True
    ) -> dict:
        """
        Predict quality scores for all images in a directory.
        
        Args:
            directory: Directory path
            extensions: List of image extensions to process
            recursive: Whether to search recursively
        
        Returns:
            Dictionary mapping image paths to quality scores
        """
        dir_path = Path(directory)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            if recursive:
                image_paths.extend(dir_path.rglob(f'*{ext}'))
            else:
                image_paths.extend(dir_path.glob(f'*{ext}'))
        
        image_paths = [str(p) for p in image_paths]
        
        print(f"Found {len(image_paths)} images")
        
        # Predict
        scores = self.predict_batch(image_paths, show_progress=True)
        
        # Create results dictionary
        results = {path: float(score) for path, score in zip(image_paths, scores)}
        
        return results
    
    def get_quality_category(self, score: float) -> str:
        """
        Get quality category from score.
        
        Args:
            score: Quality score (0-1)
        
        Returns:
            Quality category string
        """
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Bad"
