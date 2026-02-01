"""Model utility functions."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        device: Device to load model to
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint


def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def freeze_layers(model: nn.Module, freeze_until: Optional[str] = None) -> None:
    """
    Freeze model layers.
    
    Args:
        model: PyTorch model
        freeze_until: Freeze layers until this layer name (inclusive)
                     If None, freezes all layers
    """
    freeze_all = freeze_until is None
    
    for name, param in model.named_parameters():
        if freeze_all:
            param.requires_grad = False
        else:
            param.requires_grad = False
            if freeze_until in name:
                break
    
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in model.parameters())
    print(f"Frozen {frozen_params}/{total_params} parameter groups")


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze all model layers.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights using He initialization.
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> None:
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    from torch.nn import Module
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "num_params": sum(p.numel() for p in module.parameters())
            }
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    device = next(model.parameters()).device
    summary = {}
    hooks = []
    
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.zeros(1, *input_size).to(device)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print("=" * 80)
    print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 80)
    
    total_params = 0
    for layer, info in summary.items():
        total_params += info["num_params"]
        print(f"{layer:<30} {str(info['output_shape']):<25} {info['num_params']:<15}")
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print("=" * 80)
