"""Learning rate scheduler utilities."""

import torch.optim as optim
from typing import Dict, Any


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau', 'exponential')
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler parameters
    
    Returns:
        Learning rate scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            **kwargs
        )
    elif scheduler_name == 'step':
        step_size = kwargs.pop('step_size', 30)
        gamma = kwargs.pop('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == 'plateau':
        mode = kwargs.pop('mode', 'min')
        patience = kwargs.pop('patience', 10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            **kwargs
        )
    elif scheduler_name == 'exponential':
        gamma = kwargs.pop('gamma', 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == 'multistep':
        milestones = kwargs.pop('milestones', [30, 60, 90])
        gamma = kwargs.pop('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler
