# Downscaler Training Script

"""
Main training script for Downscaler model using PyTorch Lightning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from downscaler.model import DownscalerNet, DownscalerConfig, DownscalerLightningModule

logger = logging.getLogger(__name__)


def train_downscaler(config: DictConfig):
    """Main training function for Downscaler"""
    
    L.seed_everything(config.training.seed, workers=True)
    
    # Model
    model_config = DownscalerConfig(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        activation=config.model.activation,
        quantiles=config.model.quantiles
    )
    
    model = DownscalerLightningModule(config)
    
    # TODO: Create data module
    # data_module = DownscalerDataModule(config)
    # data_module.setup()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename='downscaler-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.training.patience,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name='downscaler'
    )
    
    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project='weather-intelligence',
            name=config.training.experiment_name,
            config=OmegaConf.to_container(config, resolve=True)
        )
        loggers = [logger, wandb_logger]
    else:
        loggers = logger
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        num_sanity_val_steps=2
    )
    
    # Train (with data module when available)
    # trainer.fit(model, data_module)
    # trainer.test(model, data_module)
    
    logger.info("Downscaler training setup complete!")
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


@hydra.main(config_path="../configs", config_name="downscaler", version_base="1.3")
def main(config: DictConfig):
    logger.info(f"Starting Downscaler training with config:\n{OmegaConf.to_yaml(config)}")
    
    model = train_downscaler(config)
    
    logger.info("Downscaler training setup completed!")
    return model


if __name__ == "__main__":
    main()