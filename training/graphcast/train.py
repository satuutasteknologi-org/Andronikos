# GraphCast Training Script

"""
Main training script for GraphCast model using PyTorch Lightning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from graphcast.model import GraphCast, GraphCastConfig
from graphcast.data import GraphCastDataConfig, GraphCastDataModule

logger = logging.getLogger(__name__)


class GraphCastLightningModule(L.LightningModule):
    """PyTorch Lightning module for GraphCast training"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Model configuration
        model_config = GraphCastConfig(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            surface_vars=config.model.surface_vars,
            atmosphere_vars=config.model.atmosphere_vars,
            pressure_levels=config.model.pressure_levels,
            dropout=config.model.dropout
        )
        
        self.model = GraphCast(model_config)
        
        # Loss weights
        self.loss_weights = config.training.loss_weights
        self.spectral_weight = config.training.spectral_weight
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, surface, atmosphere, lat_lon, temporal, edge_index):
        return self.model(surface, atmosphere, lat_lon, temporal, edge_index)
    
    def training_step(self, batch, batch_idx):
        # Prepare inputs
        surface = batch['surface']
        atmosphere = batch['atmosphere']
        lat_lon = batch['lat_lon']
        temporal = batch['temporal']
        target_surface = batch['target_surface']
        
        # Create edge index for graph (simplified - full connectivity or KNN)
        N = surface.shape[1]  # num_nodes
        edge_index = self._create_edge_index(surface.shape[1], surface.shape[2])
        edge_index = edge_index.to(surface.device)
        
        # Forward pass
        outputs = self(surface, batch['atmosphere'], batch['lat_lon'], 
                       batch['temporal'], edge_index)
        
        # Compute loss
        loss = self._compute_loss(outputs, batch['target_surface'])
        
        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        surface = batch['surface']
        atmosphere = batch['atmosphere']
        lat_lon = batch['lat_lon']
        temporal = batch['temporal']
        target_surface = batch['target_surface']
        
        N = surface.shape[1]
        edge_index = self._create_edge_index(N).to(surface.device)
        
        outputs = self(surface, batch['atmosphere'], batch['lat_lon'], 
                       batch['temporal'], edge_index)
        
        loss = self._compute_loss(outputs, batch['target_surface'])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Per-variable metrics
        self._log_variable_metrics(outputs, batch['target_surface'], 'val')
        
        return loss
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                      targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-task loss with spectral regularization"""
        
        # MSE loss for each variable
        mse_loss = 0.0
        for i, var in enumerate(self.model.decoder.output_vars):
            if var in outputs:
                pred = outputs[var]
                # Target shape: [B, T, lat, lon, vars]
                target = targets[..., i]  # Need to match dimensions
                
                if pred.shape != target.shape:
                    # Handle shape mismatch
                    target = target.reshape(pred.shape)
                
                var_loss = F.mse_loss(pred, target)
                mse_loss += var_loss
        
        # Spectral loss for anti-blurring
        spectral_loss = self._spectral_loss(outputs, targets)
        
        total_loss = mse_loss + self.spectral_weight * spectral_loss
        
        return total_loss
    
    def _spectral_loss(self, outputs: Dict[str, torch.Tensor], 
                       targets: torch.Tensor) -> torch.Tensor:
        """Spectral variance penalty to preserve sharp features"""
        spectral_loss = 0.0
        count = 0
        
        for var in ['temperature', 'precipitation']:
            if var in outputs:
                pred = outputs[var]
                target = targets[..., 0]  # Simplified - need proper indexing
                
                # Compute FFT
                pred_fft = torch.fft.rfft2(pred)
                target_fft = torch.fft.rfft2(targets[..., 0])  # Simplified
                
                # Spectral variance penalty
                var_pred = torch.var(torch.abs(pred_fft))
                var_target = torch.var(torch.abs(target_fft))
                spectral_loss += (var_pred - var_target).pow(2)
                count += 1
        
        return spectral_loss / max(count, 1)
    
    def _create_edge_index(self, H: int, W: int) -> torch.Tensor:
        """Create 4-connected grid edge index"""
        edges = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                # Right neighbor
                if j + 1 < W:
                    edges.append([idx, i * W + (j + 1)])
                # Down neighbor
                if i + 1 < H:
                    edges.append([idx, (i + 1) * W + j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _log_variable_metrics(self, outputs: Dict[str, torch.Tensor],
                              targets: torch.Tensor, prefix: str):
        """Log per-variable metrics"""
        for i, var in enumerate(self.model.decoder.output_vars):
            if var in outputs:
                pred = outputs[var]
                target = targets[..., i] if targets.dim() > 4 else targets
                
                mae = F.l1_loss(outputs[var], targets[..., i]).item()
                mse = F.mse_loss(outputs[var], targets[..., i]).item()
                
                self.log(f'{prefix}_mae_{var}', mae, on_epoch=True)
                self.log(f'{prefix}_mse_{var}', mse, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing with warmup
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_train_epoch_end(self):
        self.train_losses.append(self.trainer.callback_metrics.get('train_loss', 0))
    
    def on_validation_epoch_end(self):
        self.val_losses.append(self.trainer.callback_metrics.get('val_loss', 0))


def train_graphcast(config: DictConfig):
    """Main training function"""
    
    # Set seed for reproducibility
    L.seed_everything(config.training.seed, workers=True)
    
    # Data module
    data_config = GraphCastDataConfig(
        era5_path=config.data.era5_path,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        history_steps=config.data.history_steps,
        forecast_steps=config.data.forecast_steps,
    )
    
    data_module = GraphCastDataModule(config=data_config)
    data_module.setup()
    
    # Model
    model = GraphCastLightningModule(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename='graphcast-{epoch:02d}-{val_loss:.4f}',
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
        name='graphcast'
    )
    
    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project='weather-intelligence',
            name=config.training.experiment_name,
            config=OmegaConf.to_container(config, resolve=True)
        )
        logger = [logger, wandb_logger]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        num_sanity_val_steps=2
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    return model


@hydra.main(config_path="../configs", config_name="graphcast", version_base="1.3")
def main(config: DictConfig):
    logger.info(f"Starting GraphCast training with config:\n{OmegaConf.to_yaml(config)}")
    
    model = train_graphcast(config)
    
    logger.info("Training completed successfully!")
    return model


if __name__ == "__main__":
    main()