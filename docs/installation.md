# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/AvazbekHasanov/endoscopic-iqa-project.git
cd endoscopic-iqa-project
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch (adjust based on your CUDA version)
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### 4. Install Package

```bash
# Development mode (recommended for development)
pip install -e .

# Or standard installation
pip install .
```

### 5. Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python docs/examples/basic_usage.py
```

## GPU Setup

### Check CUDA Availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Install CUDA (if needed)

Visit [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) for installation instructions.

## Troubleshooting

### ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

### CUDA Out of Memory

Reduce batch size in configuration files or use CPU mode.

### Albumentations Issues

```bash
pip install --upgrade albumentations
```

## Optional Dependencies

### For Development

```bash
pip install -e ".[dev]"
```

### For API Server

```bash
pip install fastapi uvicorn
```

### For Visualization

```bash
pip install matplotlib seaborn plotly
```

## Docker Installation (Coming Soon)

Docker support will be added in future releases for easier deployment.

## Next Steps

After installation, check out:
- [Usage Guide](usage.md)
- [API Reference](api_reference.md)
- [Examples](examples/basic_usage.py)
