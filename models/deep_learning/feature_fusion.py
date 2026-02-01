"""
Feature fusion modules for combining multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FeatureFusion(nn.Module):
    """
    Simple feature fusion module.
    Combines features from different layers through concatenation and convolution.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        fusion_type: str = 'concat'
    ):
        """
        Initialize feature fusion.
        
        Args:
            in_channels_list: List of input channel sizes
            out_channels: Output channel size
            fusion_type: Type of fusion ('concat', 'add', 'attention')
        """
        super(FeatureFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            total_channels = sum(in_channels_list)
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'add':
            # Project all inputs to same channel size
            self.projections = nn.ModuleList([
                nn.Conv2d(in_ch, out_channels, 1)
                for in_ch in in_channels_list
            ])
            self.fusion_conv = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.projections = nn.ModuleList([
                nn.Conv2d(in_ch, out_channels, 1)
                for in_ch in in_channels_list
            ])
            self.attention_weights = nn.Sequential(
                nn.Conv2d(out_channels * len(in_channels_list), len(in_channels_list), 1),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: List of feature tensors with potentially different spatial sizes
        
        Returns:
            Fused features (B, out_channels, H, W)
        """
        # Get target size from the first feature
        target_size = features[0].shape[2:]
        
        # Resize all features to target size
        resized_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            resized_features.append(feat)
        
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            concatenated = torch.cat(resized_features, dim=1)
            output = self.fusion_conv(concatenated)
            
        elif self.fusion_type == 'add':
            # Project and add
            projected = [
                proj(feat) for proj, feat in zip(self.projections, resized_features)
            ]
            output = sum(projected)
            output = self.fusion_conv(output)
            
        elif self.fusion_type == 'attention':
            # Project all features
            projected = [
                proj(feat) for proj, feat in zip(self.projections, resized_features)
            ]
            
            # Compute attention weights
            stacked = torch.cat(projected, dim=1)
            weights = self.attention_weights(stacked)  # (B, num_features, H, W)
            
            # Apply attention weights
            output = sum([
                proj * weights[:, i:i+1, :, :]
                for i, proj in enumerate(projected)
            ])
        
        return output


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion with pyramid pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: List[int] = [1, 2, 4, 8]
    ):
        """
        Initialize multi-scale fusion.
        
        Args:
            in_channels: Input channel size
            out_channels: Output channel size
            scales: List of pooling scales
        """
        super(MultiScaleFusion, self).__init__()
        
        self.scales = scales
        
        # Pooling branches
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in scales
        ])
        
        # Channel reduction for each scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(scales), 1),
                nn.BatchNorm2d(in_channels // len(scales)),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Multi-scale fused features (B, out_channels, H, W)
        """
        h, w = x.shape[2:]
        
        # Apply multi-scale pooling
        pooled_features = []
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            reduced = conv(pooled)
            upsampled = F.interpolate(
                reduced, size=(h, w),
                mode='bilinear', align_corners=False
            )
            pooled_features.append(upsampled)
        
        # Concatenate multi-scale features
        concatenated = torch.cat(pooled_features, dim=1)
        
        # Fusion
        output = self.fusion(concatenated)
        
        return output


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion with learnable fusion weights.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int
    ):
        """
        Initialize adaptive fusion.
        
        Args:
            in_channels_list: List of input channel sizes
            out_channels: Output channel size
        """
        super(AdaptiveFeatureFusion, self).__init__()
        
        self.num_inputs = len(in_channels_list)
        
        # Project all inputs to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(
            torch.ones(self.num_inputs) / self.num_inputs
        )
        
        # Output refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: List of feature tensors
        
        Returns:
            Adaptively fused features (B, out_channels, H, W)
        """
        # Get target size
        target_size = features[0].shape[2:]
        
        # Project and resize all features
        projected = []
        for proj, feat in zip(self.projections, features):
            feat_proj = proj(feat)
            if feat_proj.shape[2:] != target_size:
                feat_proj = F.interpolate(
                    feat_proj, size=target_size,
                    mode='bilinear', align_corners=False
                )
            projected.append(feat_proj)
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted sum
        output = sum([w * feat for w, feat in zip(weights, projected)])
        
        # Refinement
        output = self.refinement(output)
        
        return output
