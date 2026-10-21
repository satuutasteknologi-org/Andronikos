# Downscaler Model Definition (PyTorch)

"""
Downscaler model for regional weather downscaling.
Takes coarse global forecast + static features -> high-resolution local forecast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DownscalerConfig:
    """Downscaler model configuration"""
    # Input features (14 total)
    # 2: lat/lon (sin/cos)
    # 1: lead_time (hours)
    # 6: coarse variables (T, RH, precip, U, V, P)
    # 1: elevation
    # 1: land_mask
    # 1: month (sin/cos)
    # 1: hour (sin/cos)
    # 1: sin_month
    input_dim: int = 14
    
    # Output: 6 variables * 4 quantiles = 24
    # Variables: T, RH, precip, U, V, P
    # Each: p10, p50, p90, prob_rain (for precip)
    output_dim: int = 24
    num_variables: int = 6
    num_quantiles: int = 4
    
    # Architecture
    hidden_dims: List[int] = None
    dropout: float = 0.1
    activation: str = "silu"
    
    # Quantile regression
    quantiles: List[float] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 512, 256]
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class QuantileLoss(nn.Module):
    """Quantile regression loss (pinball loss)"""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [B, num_quantiles] - predicted quantiles
            targets: [B] - observed values
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:, i]
            loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()


class DownscalerNet(nn.Module):
    """Downscaler neural network"""
    
    def __init__(self, config: DownscalerConfig):
        super().__init__()
        self.config = config
        
        if config.hidden_dims is None:
            config.hidden_dims = [256, 512, 512, 256]
        if config.quantiles is None:
            config.quantiles = [0.1, 0.5, 0.9]
        
        # Build MLP
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            if config.activation == "silu":
                layers.append(nn.SiLU())
            elif config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads for each variable
        # 6 variables * 4 quantiles = 24 outputs
        self.output_head = nn.Linear(prev_dim, config.output_dim)
        
        # Quantile loss
        self.quantile_loss = QuantileLoss(config.quantiles)
        
        # Variable mapping
        self.variables = ['temperature', 'humidity', 'precipitation', 
                          'wind_u', 'wind_v', 'pressure']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 14] - input features
        Returns:
            [B, 24] - quantile predictions for 6 variables
        """
        features = self.backbone(x)
        outputs = self.output_head(features)  # [B, 24]
        return outputs
    
    def predict_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get quantile predictions as dictionary
        
        Returns:
            Dict with keys: temperature, humidity, precipitation, wind_u, wind_v, pressure
            Each value: [B, 4] (p10, p50, p90, prob_rain)
        """
        outputs = self.forward(x)  # [B, 24]
        B = outputs.shape[0]
        
        results = {}
        for i, var in enumerate(self.variables):
            start = i * 4
            end = start + 4
            results[var] = outputs[:, start:end]
        
        return results
    
    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss
        
        Args:
            preds: [B, 24] - model outputs
            targets: [B, 6] - observed values for 6 variables
        """
        total_loss = 0.0
        quantiles = [0.1, 0.5, 0.9]
        
        for i in range(6):  # 6 variables
            var_preds = preds[:, i*4:(i+1)*4]  # [B, 4]
            var_targets = targets[:, i]  # [B]
            
            # Quantile loss for first 3 outputs
            for q_idx, q in enumerate([0.1, 0.5, 0.9]):
                errors = targets[:, i] - preds[:, i*4 + q_idx]
                loss = torch.max(q * errors, (q - 1) * errors)
                total_loss += loss.mean()
            
            # Probability of rain (4th output for precipitation)
            if i == 2:  # precipitation index
                prob_rain = torch.sigmoid(preds[:, i*4 + 3])
                rain_obs = (targets[:, i] > 0.1).float()
                bce_loss = F.binary_cross_entropy(prob_rain, rain_obs)
                total_loss += bce_loss
        
        return total_loss


class DownscalerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Downscaler training"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        model_config = DownscalerConfig(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
            activation=config.model.activation,
            quantiles=config.model.quantiles
        )
        
        self.model = DownscalerNet(model_config)
        self.criterion = nn.MSELoss()  # Fallback
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['features']
        targets = batch['targets']
        
        preds = self(x)
        loss = self.model.loss(preds, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        targets = batch['targets']
        
        preds = self(x)
        loss = self.model.loss(preds, targets)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def train_downscaler(config: DictConfig):
    """Main training function for downscaler"""
    
    L.seed_everything(config.training.seed, workers=True)
    
    # Data
    # TODO: Implement DownscalerDataModule
    
    # Model
    model = DownscalerLightningModule(config)
    
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
        EarlyStopping(monitor='val_loss', patience=config.training.patience, mode='min'),
        LearningRateMonitor(logging_interval='step')
    ]
    
    logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name='downscaler'
    )
    
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(config.training.log_dir, name='downscaler'),
        log_every_n_steps=config.training.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches
    )
    
    # trainer.fit(model)  # TODO: add data module
    # trainer.test(model)
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = DownscalerConfig()
    model = DownscalerNet(config)
    
    print(f"Downscaler model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward
    x = torch.randn(32, 14)
    outputs = model(x)
    print(f"Output shape: {outputs.shape}")  # Should be [32, 24]
    
    # Test quantile prediction
    quantiles = model.predict_quantiles(x)
    for var, q in quantiles.items():
        print(f"  {var}: {q.shape}")  # [B, 4]
    
    # Test loss
    targets = torch.randn(32, 6)
    loss = model.loss(outputs, targets)
    print(f"Loss: {loss.item():.4f}")