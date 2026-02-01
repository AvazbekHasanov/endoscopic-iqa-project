"""
Main CNN architecture for image quality assessment.
Lightweight models designed for real-time endoscopic image quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .attention import SpatialAttention, ChannelAttention, CBAM
from .feature_fusion import FeatureFusion, MultiScaleFusion


class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_attention: bool = False
    ):
        """
        Initialize conv block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Stride
            padding: Padding
            use_attention: Whether to use CBAM attention
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = CBAM(out_channels) if use_attention else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        """Initialize depthwise separable conv."""
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride,
            padding=1, groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightweightIQAModel(nn.Module):
    """
    Lightweight CNN model for real-time IQA.
    Inspired by MobileNet architecture for efficiency.
    Target: <50MB model size, <100ms inference time.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        use_attention: bool = True
    ):
        """
        Initialize lightweight IQA model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for quality score)
            base_channels: Base number of channels
            use_attention: Whether to use attention mechanisms
        """
        super(LightweightIQAModel, self).__init__()
        
        # Initial conv layer
        self.initial = ConvBlock(
            in_channels, base_channels,
            kernel_size=3, stride=2, padding=1
        )
        
        # Encoder blocks with downsampling
        self.encoder1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparableConv(base_channels * 2, base_channels * 2)
        )
        
        self.encoder2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, stride=2),
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4)
        )
        
        self.encoder3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 8, stride=2),
            DepthwiseSeparableConv(base_channels * 8, base_channels * 8)
        )
        
        # Attention modules
        self.use_attention = use_attention
        if use_attention:
            self.attention1 = CBAM(base_channels * 2)
            self.attention2 = CBAM(base_channels * 4)
            self.attention3 = CBAM(base_channels * 8)
        
        # Multi-scale feature fusion
        self.fusion = MultiScaleFusion(
            in_channels=base_channels * 8,
            out_channels=base_channels * 8
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Quality score (B, 1)
        """
        # Initial convolution
        x = self.initial(x)  # (B, 32, H/2, W/2)
        
        # Encoder stages
        x1 = self.encoder1(x)  # (B, 64, H/4, W/4)
        if self.use_attention:
            x1 = self.attention1(x1)
        
        x2 = self.encoder2(x1)  # (B, 128, H/8, W/8)
        if self.use_attention:
            x2 = self.attention2(x2)
        
        x3 = self.encoder3(x2)  # (B, 256, H/16, W/16)
        if self.use_attention:
            x3 = self.attention3(x3)
        
        # Multi-scale fusion
        features = self.fusion(x3)
        
        # Global pooling
        pooled = self.global_pool(features)  # (B, 256, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 256)
        
        # Regression
        score = self.regressor(pooled)  # (B, 1)
        
        # Apply sigmoid to get score in [0, 1]
        score = torch.sigmoid(score)
        
        return score


class IQAModel(nn.Module):
    """
    Full IQA model with multi-scale feature extraction and fusion.
    More accurate but slightly larger than LightweightIQAModel.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        use_multi_scale_fusion: bool = True
    ):
        """
        Initialize IQA model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels
            use_multi_scale_fusion: Whether to use multi-scale fusion
        """
        super(IQAModel, self).__init__()
        
        self.use_multi_scale_fusion = use_multi_scale_fusion
        
        # Encoder backbone
        self.conv1 = ConvBlock(in_channels, base_channels, stride=2)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, stride=2, use_attention=True)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, use_attention=True)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2, use_attention=True)
        
        # Multi-scale feature fusion
        if use_multi_scale_fusion:
            self.feature_fusion = FeatureFusion(
                in_channels_list=[base_channels * 2, base_channels * 4, base_channels * 8],
                out_channels=base_channels * 8,
                fusion_type='attention'
            )
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Regression head
        feature_dim = base_channels * 8 * 2  # Concatenate avg and max pooling
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 4),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Quality score (B, 1)
        """
        # Feature extraction
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        
        # Multi-scale fusion
        if self.use_multi_scale_fusion:
            features = self.feature_fusion([f2, f3, f4])
        else:
            features = f4
        
        # Global pooling (both avg and max)
        avg_pooled = self.global_pool(features).view(features.size(0), -1)
        max_pooled = self.global_max_pool(features).view(features.size(0), -1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Regression
        score = self.regressor(pooled)
        
        # Apply sigmoid to get score in [0, 1]
        score = torch.sigmoid(score)
        
        return score


def get_model(
    model_type: str = 'lightweight',
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Get IQA model by type.
    
    Args:
        model_type: Type of model ('lightweight' or 'full')
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments for model
    
    Returns:
        IQA model
    """
    if model_type == 'lightweight':
        model = LightweightIQAModel(**kwargs)
    elif model_type == 'full':
        model = IQAModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained:
        # TODO: Implement pretrained weight loading
        print("Warning: Pretrained weights not implemented yet")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
