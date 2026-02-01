"""
Main training script for IQA model.
"""

import torch
import yaml
import argparse
from pathlib import Path
import random
import numpy as np

from data import create_dataloaders
from data.augmentation import get_augmentation_pipeline
from data.synthetic_degradation import SyntheticDegradation
from models.deep_learning import get_model
from training import IQATrainer
from training.losses import MSELoss, L1Loss, CombinedLoss
from training.utils import get_optimizer, get_scheduler


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_loss_function(config: dict):
    """Get loss function based on configuration."""
    loss_config = config['training']['loss']
    loss_name = loss_config['name']
    
    if loss_name == 'mse':
        return MSELoss()
    elif loss_name == 'l1':
        return L1Loss()
    elif loss_name == 'combined':
        return CombinedLoss(
            use_mse=True,
            use_l1=True,
            mse_weight=loss_config.get('mse_weight', 1.0),
            l1_weight=loss_config.get('l1_weight', 0.5)
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_model(config_path: str, resume_from: str = None):
    """
    Train IQA model.
    
    Args:
        config_path: Path to configuration file
        resume_from: Optional checkpoint path to resume from
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Set device
    device = config.get('device', 'cuda')
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output']['output_dir']) / config['output']['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Configuration saved to {config_save_path}")
    
    # Get degradation function
    degradation = None
    if config['data'].get('use_degradation', False):
        degradation = SyntheticDegradation(
            degradation_types=config['data'].get('degradation_types'),
            severity_range=tuple(config['data'].get('severity_range', [0.1, 0.9]))
        )
        print("Synthetic degradation enabled")
    
    # Get augmentation pipelines
    train_transform = None
    val_transform = None
    
    if config['data'].get('use_augmentation', True):
        aug_mode = config['data'].get('augmentation_mode', 'endoscopic')
        train_transform = get_augmentation_pipeline(
            mode='train',
            image_size=tuple(config['data']['image_size'])
        )
        val_transform = get_augmentation_pipeline(
            mode='val',
            image_size=tuple(config['data']['image_size'])
        )
        print(f"Augmentation enabled: {aug_mode}")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        train_transform=train_transform,
        val_transform=val_transform,
        degradation=degradation
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_type=config['model']['type'],
        in_channels=3,
        num_classes=config['model']['num_classes'],
        base_channels=config['model'].get('base_channels', 32),
        use_attention=config['model'].get('use_attention', True)
    )
    
    # Count parameters
    from models.deep_learning.iqa_model import count_parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    from models.utils import get_model_size
    model_size = get_model_size(model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Create loss function
    criterion = get_loss_function(config)
    print(f"Loss function: {config['training']['loss']['name']}")
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer']['name'],
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer'].get('weight_decay', 0.0001)
    )
    print(f"Optimizer: {config['training']['optimizer']['name']}")
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training']['scheduler']['name'],
        num_epochs=config['training']['num_epochs']
    )
    print(f"Scheduler: {config['training']['scheduler']['name']}")
    
    # Create trainer
    trainer = IQATrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        output_dir=str(output_dir),
        log_interval=config['training'].get('log_interval', 10)
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience')
    )
    
    print("\nTraining completed!")
    
    # Plot training history
    try:
        from evaluation.visualization import plot_training_history
        plot_path = output_dir / 'training_history.png'
        plot_training_history(history, save_path=str(plot_path))
        print(f"Training history plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not plot training history: {e}")
    
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train IQA model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    train_model(args.config, args.resume)


if __name__ == '__main__':
    main()
