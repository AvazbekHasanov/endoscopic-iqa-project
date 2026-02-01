"""
Attention mechanisms for clinical-aware image quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    Focuses on diagnostically important spatial regions.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention.
        
        Args:
            kernel_size: Convolution kernel size
        """
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Compute attention map
        attention = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention module.
    Adaptively weights feature channels.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize channel attention.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()
        
        # Global pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Compute attention weights
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Apply attention
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        """
        Initialize CBAM.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Spatial attention kernel size
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class ClinicalAttention(nn.Module):
    """
    Clinical-aware attention mechanism.
    Specifically designed for endoscopic images to focus on
    diagnostically important regions.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_regions: int = 4
    ):
        """
        Initialize clinical attention.
        
        Args:
            in_channels: Number of input channels
            num_regions: Number of clinical regions to focus on
        """
        super(ClinicalAttention, self).__init__()
        
        self.num_regions = num_regions
        
        # Region proposal network
        self.region_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_regions, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature refinement per region
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_regions)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Clinically attended features (B, C, H, W)
        """
        # Compute region attention maps
        region_maps = self.region_conv(x)  # (B, num_regions, H, W)
        
        # Apply region-specific refinement
        refined_features = []
        for i in range(self.num_regions):
            # Get region attention map
            region_attn = region_maps[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # Apply attention
            attended = x * region_attn
            
            # Refine features
            refined = self.refinement[i](attended)
            refined_features.append(refined)
        
        # Combine all refined features
        output = sum(refined_features)
        
        return output
