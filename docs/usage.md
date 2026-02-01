# Usage Guide

## Quick Start

### 1. Assess Single Image Quality

```python
from inference.predictor import IQAPredictor
from models.deep_learning import get_model

# Create model
model = get_model(model_type='lightweight')

# Initialize predictor
predictor = IQAPredictor(model=model)

# Predict quality
score = predictor.predict('path/to/image.jpg')
print(f"Quality Score: {score:.3f}")
```

### 2. Use Traditional Metrics

```python
from models.traditional.traditional_iqa import TraditionalIQA
import cv2

# Initialize IQA
iqa = TraditionalIQA()

# Load image
image = cv2.imread('path/to/image.jpg')

# Compute metrics
metrics = iqa.compute_all_metrics(image)
print(metrics)
```

### 3. Run Interactive Demo

```bash
streamlit run inference/real_time_demo.py
```

## Training a Model

### Prepare Your Data

1. Organize images in a directory structure:
```
data/datasets/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

2. Create annotation file (optional):
```json
[
    {"image_path": "train/image1.jpg", "quality_score": 0.85},
    {"image_path": "train/image2.jpg", "quality_score": 0.72}
]
```

### Configure Training

Edit `configs/training_config.yaml`:

```yaml
data:
  data_dir: 'data/datasets'
  batch_size: 32
  
model:
  type: 'lightweight'
  base_channels: 32

training:
  num_epochs: 100
  learning_rate: 0.001
```

### Start Training

```bash
python training/train.py --config configs/training_config.yaml
```

### Resume Training

```bash
python training/train.py --config configs/training_config.yaml --resume outputs/checkpoint.pth
```

## Evaluation

### Evaluate Model

```python
from evaluation.evaluator import IQAEvaluator
from models.deep_learning import get_model
from data import create_dataloaders

# Load model
model = get_model(model_type='lightweight')
# ... load checkpoint ...

# Create evaluator
evaluator = IQAEvaluator(model=model)

# Create test loader
data_loaders = create_dataloaders('data/datasets')
test_loader = data_loaders['test']

# Evaluate
metrics = evaluator.evaluate(test_loader)
```

### Visualize Results

```python
from evaluation.visualization import plot_scatter, plot_training_history

# Plot predictions vs ground truth
plot_scatter(
    predictions, 
    targets, 
    save_path='results/scatter.png',
    plcc=0.85,
    srcc=0.83
)

# Plot training history
plot_training_history(history, save_path='results/history.png')
```

## Batch Processing

### Process Directory of Images

```python
from inference.predictor import IQAPredictor

predictor = IQAPredictor(model_path='models/best_model.pth')

# Process all images in directory
results = predictor.predict_directory('path/to/images/')

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
```

### Process Video

```python
# Assess video quality frame by frame
scores = predictor.predict_video(
    'path/to/video.mp4',
    output_path='output_video.mp4',
    sample_rate=5  # Process every 5th frame
)

print(f"Average quality: {np.mean(scores):.3f}")
```

## Advanced Usage

### Custom Degradation

```python
from data.synthetic_degradation import SyntheticDegradation

degrader = SyntheticDegradation(
    degradation_types=['motion_blur', 'gaussian_noise'],
    severity_range=(0.2, 0.8)
)

# Apply to image
degraded, quality = degrader.apply_random_degradation(image)
```

### Custom Model Architecture

```python
from models.deep_learning import IQAModel

# Create custom model
model = IQAModel(
    in_channels=3,
    base_channels=64,
    use_multi_scale_fusion=True
)
```

### Cross-Validation

```python
from evaluation.evaluator import IQAEvaluator

# Create fold data loaders
fold_loaders = [loader1, loader2, loader3, loader4, loader5]

# Run cross-validation
cv_results = evaluator.cross_validate(
    fold_loaders,
    fold_names=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
)
```

## Tips and Best Practices

1. **Use GPU**: Enable CUDA for faster training and inference
2. **Batch Size**: Adjust based on available GPU memory
3. **Data Augmentation**: Use appropriate augmentation for medical images
4. **Early Stopping**: Enable to prevent overfitting
5. **Learning Rate**: Start with 0.001 and adjust based on training curves
6. **Model Size**: Use lightweight model for real-time applications
7. **Validation**: Always validate on separate test set

## Troubleshooting

### Low Correlation Scores

- Increase training data
- Use more aggressive data augmentation
- Try different model architectures
- Tune hyperparameters

### Slow Inference

- Use lightweight model
- Enable GPU acceleration
- Reduce image size
- Use batch processing

### Memory Issues

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

## Next Steps

- Check [API Reference](api_reference.md) for detailed documentation
- See [Examples](examples/) for more usage patterns
- Read source code for advanced customization
