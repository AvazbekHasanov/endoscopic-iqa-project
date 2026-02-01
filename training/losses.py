"""Loss functions for IQA training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error loss for regression."""
    
    def __init__(self):
        """Initialize MSE loss."""
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            predictions: Predicted quality scores (B, 1)
            targets: Ground truth scores (B, 1)
        
        Returns:
            MSE loss value
        """
        return self.mse(predictions, targets)


class L1Loss(nn.Module):
    """L1 (MAE) loss for regression."""
    
    def __init__(self):
        """Initialize L1 loss."""
        super(L1Loss, self).__init__()
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            predictions: Predicted quality scores (B, 1)
            targets: Ground truth scores (B, 1)
        
        Returns:
            L1 loss value
        """
        return self.l1(predictions, targets)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss (Huber loss)."""
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize Smooth L1 loss.
        
        Args:
            beta: Threshold for switching between L1 and L2
        """
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss.
        
        Args:
            predictions: Predicted quality scores
            targets: Ground truth scores
        
        Returns:
            Smooth L1 loss value
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        use_mse: bool = True,
        use_l1: bool = True,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5
    ):
        """
        Initialize combined loss.
        
        Args:
            use_mse: Whether to use MSE loss
            use_l1: Whether to use L1 loss
            mse_weight: Weight for MSE loss
            l1_weight: Weight for L1 loss
        """
        super(CombinedLoss, self).__init__()
        
        self.use_mse = use_mse
        self.use_l1 = use_l1
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        
        if use_mse:
            self.mse_loss = nn.MSELoss()
        if use_l1:
            self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted quality scores
            targets: Ground truth scores
        
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        if self.use_mse:
            mse = self.mse_loss(predictions, targets)
            total_loss += self.mse_weight * mse
        
        if self.use_l1:
            l1 = self.l1_loss(predictions, targets)
            total_loss += self.l1_weight * l1
        
        return total_loss


class RankingLoss(nn.Module):
    """
    Ranking loss to preserve relative quality ordering.
    Useful when absolute scores might be noisy but relative ordering is reliable.
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Initialize ranking loss.
        
        Args:
            margin: Margin for ranking loss
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss.
        
        Args:
            predictions: Predicted quality scores (B,)
            targets: Ground truth scores (B,)
        
        Returns:
            Ranking loss value
        """
        # Create pairs of samples
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute pairwise differences
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        
        # Determine correct ordering: target_diff > 0 means first is better
        correct_order = (target_diff > 0).float()
        
        # Loss: penalize when prediction doesn't follow target ordering
        loss = F.relu(self.margin - pred_diff * correct_order)
        
        # Average over pairs
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features.
    Can be combined with regression loss for better feature learning.
    """
    
    def __init__(self, feature_extractor: nn.Module):
        """
        Initialize perceptual loss.
        
        Args:
            feature_extractor: Pre-trained model for feature extraction
        """
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            predictions: Predicted quality scores (not used directly)
            targets: Target quality scores (not used directly)
            images: Input images for feature extraction
        
        Returns:
            Perceptual loss value
        """
        # Extract features
        features = self.feature_extractor(images)
        
        # Compute loss based on feature variance or other criteria
        # This is a simplified version
        loss = -features.var()  # Encourage diverse features
        
        return loss
