# API Reference

## Data Module

### `data.EndoscopicDataset`

Dataset class for loading endoscopic images with quality scores.

```python
from data import EndoscopicDataset

dataset = EndoscopicDataset(
    data_dir='path/to/data',
    annotation_file='annotations.json',  # Optional
    transform=None,
    degradation=None,
    mode='train',
    image_size=(224, 224)
)
```

**Parameters:**
- `data_dir` (str): Root directory containing images
- `annotation_file` (str, optional): JSON file with image paths and quality scores
- `transform` (callable, optional): Transform/augmentation function
- `degradation` (callable, optional): Degradation function
- `mode` (str): Dataset mode ('train', 'val', 'test')
- `image_size` (tuple): Target image size (height, width)

### `data.SyntheticDegradation`

Apply synthetic degradations to images.

```python
from data import SyntheticDegradation

degrader = SyntheticDegradation(
    degradation_types=['motion_blur', 'gaussian_noise'],
    severity_range=(0.1, 0.9)
)

degraded_image, quality_score = degrader.apply_random_degradation(image)
```

**Methods:**
- `apply_motion_blur(image, severity)`: Apply motion blur
- `apply_defocus_blur(image, severity)`: Apply defocus blur
- `apply_gaussian_noise(image, severity)`: Add Gaussian noise
- `apply_poisson_noise(image, severity)`: Add Poisson noise
- `apply_illumination_variation(image, severity)`: Vary illumination
- `apply_specular_reflection(image, severity)`: Add specular reflections
- `apply_color_distortion(image, severity)`: Distort colors
- `apply_random_degradation(image, base_quality)`: Apply random degradation
- `apply_multiple_degradations(image, num_degradations)`: Apply multiple degradations

## Models Module

### `models.traditional.TraditionalIQA`

Traditional (handcrafted) IQA metrics.

```python
from models.traditional import TraditionalIQA

iqa = TraditionalIQA()
metrics = iqa.compute_all_metrics(image)
quality_score = iqa.compute_quality_score(image, method='combined')
```

**Methods:**
- `compute_all_metrics(image)`: Compute all metrics
- `laplacian_variance(image)`: Blur detection using Laplacian variance
- `gradient_energy(image)`: Sharpness using gradient magnitude
- `rms_contrast(image)`: RMS contrast
- `image_entropy(image)`: Shannon entropy
- `estimate_noise(image)`: Noise level estimation
- `tenengrad(image)`: Tenengrad focus measure
- `mscn_coefficients(image)`: MSCN coefficients (BRISQUE)
- `compute_quality_score(image, method)`: Overall quality score
- `assess_blur(image, threshold)`: Check if image is blurry
- `assess_noise(image, threshold)`: Check if image is noisy

### `models.deep_learning.LightweightIQAModel`

Lightweight CNN for real-time IQA.

```python
from models.deep_learning import LightweightIQAModel

model = LightweightIQAModel(
    in_channels=3,
    num_classes=1,
    base_channels=32,
    use_attention=True
)
```

**Parameters:**
- `in_channels` (int): Number of input channels (3 for RGB)
- `num_classes` (int): Number of output classes (1 for quality score)
- `base_channels` (int): Base number of channels
- `use_attention` (bool): Whether to use attention mechanisms

### `models.deep_learning.IQAModel`

Full CNN for accurate IQA.

```python
from models.deep_learning import IQAModel

model = IQAModel(
    in_channels=3,
    num_classes=1,
    base_channels=64,
    use_multi_scale_fusion=True
)
```

### `models.deep_learning.get_model`

Factory function to create models.

```python
from models.deep_learning import get_model

model = get_model(
    model_type='lightweight',  # or 'full'
    pretrained=False,
    **kwargs
)
```

## Training Module

### `training.IQATrainer`

Trainer class for model training.

```python
from training import IQATrainer

trainer = IQATrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    scheduler=scheduler,
    output_dir='outputs',
    log_interval=10
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=15
)
```

**Methods:**
- `train_epoch(train_loader)`: Train for one epoch
- `validate(val_loader)`: Validate model
- `train(train_loader, val_loader, num_epochs, early_stopping_patience)`: Full training loop
- `save_checkpoint(epoch, val_loss, is_best)`: Save checkpoint
- `load_checkpoint(checkpoint_path)`: Load checkpoint
- `save_history()`: Save training history

### `training.losses`

Loss functions for IQA training.

```python
from training.losses import MSELoss, L1Loss, CombinedLoss

# MSE loss
criterion = MSELoss()

# Combined loss
criterion = CombinedLoss(
    use_mse=True,
    use_l1=True,
    mse_weight=1.0,
    l1_weight=0.5
)
```

## Evaluation Module

### `evaluation.IQAEvaluator`

Evaluator for comprehensive model evaluation.

```python
from evaluation import IQAEvaluator

evaluator = IQAEvaluator(
    model=model,
    device='cuda',
    output_dir='evaluation_results'
)

metrics = evaluator.evaluate(test_loader, save_predictions=True)
```

**Methods:**
- `evaluate(data_loader, save_predictions)`: Evaluate on dataset
- `cross_validate(data_loaders, fold_names)`: K-fold cross-validation
- `compare_models(models, data_loader)`: Compare multiple models
- `print_metrics(metrics)`: Print formatted metrics
- `save_results()`: Save results to file
- `save_predictions(predictions, targets, paths)`: Save predictions

### `evaluation.metrics`

Evaluation metrics functions.

```python
from evaluation.metrics import (
    compute_plcc, compute_srcc, 
    compute_rmse, compute_mae,
    compute_all_metrics
)

# Compute individual metrics
plcc, p_value = compute_plcc(predictions, targets)
srcc, p_value = compute_srcc(predictions, targets)
rmse = compute_rmse(predictions, targets)
mae = compute_mae(predictions, targets)

# Compute all metrics at once
metrics = compute_all_metrics(predictions, targets)
```

### `evaluation.visualization`

Visualization utilities.

```python
from evaluation.visualization import (
    plot_scatter,
    plot_training_history,
    plot_metric_comparison,
    plot_error_distribution
)

# Scatter plot
plot_scatter(
    predictions, 
    targets,
    title='Quality Score Comparison',
    save_path='scatter.png',
    plcc=0.85,
    srcc=0.83
)

# Training history
plot_training_history(history, save_path='history.png')

# Metric comparison
plot_metric_comparison(
    results={'Method1': metrics1, 'Method2': metrics2},
    metrics=['plcc', 'srcc', 'rmse', 'mae'],
    save_path='comparison.png'
)
```

## Inference Module

### `inference.IQAPredictor`

Predictor for IQA inference.

```python
from inference import IQAPredictor

predictor = IQAPredictor(
    model=model,
    # or model_path='path/to/checkpoint.pth',
    device='cuda',
    image_size=(224, 224),
    batch_size=32
)

# Single image
score = predictor.predict('image.jpg')
score, time_ms = predictor.predict('image.jpg', return_time=True)

# Batch prediction
scores = predictor.predict_batch(image_list, show_progress=True)

# Video processing
scores = predictor.predict_video(
    'video.mp4',
    output_path='annotated_video.mp4',
    sample_rate=5
)

# Directory processing
results = predictor.predict_directory(
    'path/to/images',
    extensions=['.jpg', '.png'],
    recursive=True
)
```

**Methods:**
- `predict(image, return_time)`: Predict quality for single image
- `predict_batch(images, show_progress)`: Predict for batch
- `predict_video(video_path, output_path, sample_rate, show_preview)`: Process video
- `predict_directory(directory, extensions, recursive)`: Process directory
- `get_quality_category(score)`: Get quality category from score

## Configuration Files

### Training Configuration (`configs/training_config.yaml`)

```yaml
model:
  type: 'lightweight'
  base_channels: 32
  use_attention: true

data:
  data_dir: 'data/datasets'
  batch_size: 32
  image_size: [224, 224]

training:
  num_epochs: 100
  optimizer:
    name: 'adamw'
    learning_rate: 0.001
  scheduler:
    name: 'cosine'
  loss:
    name: 'combined'
```

### Demo Configuration (`configs/demo_config.yaml`)

```yaml
model:
  model_path: 'models/pretrained/best_model.pth'
  device: 'cuda'

inference:
  image_size: [224, 224]
  thresholds:
    excellent: 0.8
    good: 0.6
    fair: 0.4
```

## Command Line Interface

### Training

```bash
python training/train.py --config configs/training_config.yaml
python training/train.py --config configs/training_config.yaml --resume checkpoint.pth
```

### Demo

```bash
streamlit run inference/real_time_demo.py
```

## Quality Score Interpretation

- **0.8 - 1.0**: Excellent quality
- **0.6 - 0.8**: Good quality
- **0.4 - 0.6**: Fair quality
- **0.2 - 0.4**: Poor quality
- **0.0 - 0.2**: Bad quality

## Performance Requirements

- **Processing Speed**: <100ms per frame on GPU
- **Model Size**: <50MB for deployment
- **Correlation**: PLCC > 0.80, SRCC > 0.80
- **Memory**: <2GB GPU memory during inference
