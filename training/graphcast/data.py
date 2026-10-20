# GraphCast Data Loading Module

"""
Data loading and preprocessing for GraphCast training using ERA5 data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class GraphCastDataConfig:
    """Configuration for GraphCast data loading"""
    # Data paths
    era5_path: str = "/data/era5"
    era5_single_level: str = "era5_single_level.nc"
    era5_pressure_level: str = "era5_pressure_level.nc"
    
    # Variables
    surface_vars: List[str] = None
    atmosphere_vars: List[str] = None
    pressure_levels: List[int] = None
    
    # Time
    start_date: str = "2010-01-01"
    end_date: str = "2023-12-31"
    time_step_hours: int = 1
    history_steps: int = 4
    forecast_steps: int = 1
    
    # Spatial
    lat_min: float = -11.0
    lat_max: float = 6.1
    lon_min: float = 95.0
    lon_max: float = 141.1
    resolution: float = 0.25  # ERA5 native resolution
    
    # Training
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Normalization
    use_climatology_norm: bool = True
    climatology_path: Optional[str] = None
    
    def __post_init__(self):
        if self.surface_vars is None:
            self.surface_vars = [
                't2m', 'd2m', 'tp', 'u10', 'v10', 'sp', 'tcc', 'ssrd'
            ]
        if self.atmosphere_vars is None:
            self.atmosphere_vars = [
                't', 'q', 'u', 'v', 'z', 'w'
            ]
        if self.pressure_levels is None:
            self.pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


class ERA5Dataset(Dataset):
    """ERA5 dataset for GraphCast training"""
    
    def __init__(self, config: GraphCastDataConfig, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Load data
        self._load_data()
        self._compute_indices()
        
        logger.info(f"ERA5Dataset({split}): {len(self)} samples")
    
    def _load_data(self):
        """Load ERA5 data from NetCDF files"""
        sl_path = Path(self.config.era5_path) / self.config.era5_single_level
        pl_path = Path(self.config.era5_path) / self.config.era5_pressure_level
        
        logger.info(f"Loading ERA5 data from {sl_path} and {pl_path}")
        
        # Load single-level data
        self.sl_data = xr.open_dataset(sl_path, chunks={'time': 24})
        self.sl_data = self.sl_data.sel(
            latitude=slice(self.config.lat_max, self.config.lat_min),
            longitude=slice(self.config.lon_min, self.config.lon_max),
            time=slice(self.config.start_date, self.config.end_date)
        )
        
        # Load pressure-level data
        self.pl_data = xr.open_dataset(pl_path, chunks={'time': 24})
        self.pl_data = self.pl_data.sel(
            latitude=slice(self.config.lat_max, self.config.lat_min),
            longitude=slice(self.config.lon_min, self.config.lon_max),
            level=self.config.pressure_levels,
            time=slice(self.config.start_date, self.config.end_date)
        )
        
        # Select variables
        self.sl_data = self.sl_data[self.config.surface_vars]
        self.pl_data = self.pl_data[self.config.atmosphere_vars]
        
        # Compute normalization stats if not provided
        if self.config.use_climatology_norm and self.config.climatology_path:
            self._load_climatology()
        else:
            self._compute_normalization()
    
    def _compute_normalization(self):
        """Compute mean/std for normalization"""
        logger.info("Computing normalization statistics...")
        
        self.norm_stats = {}
        
        # Surface variables
        for var in self.config.surface_vars:
            data = self.sl_data[var].values
            valid = data[~np.isnan(data)]
            self.norm_stats[var] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid))
            }
        
        # Atmosphere variables
        for var in self.config.atmosphere_vars:
            data = self.pl_data[var].values
            valid = data[~np.isnan(data)]
            self.norm_stats[var] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid))
            }
    
    def _load_climatology(self):
        """Load pre-computed climatology for normalization"""
        clim = xr.open_dataset(self.config.climatology_path)
        self.norm_stats = {}
        for var in self.config.surface_vars + self.config.atmosphere_vars:
            if var in clim:
                self.norm_stats[var] = {
                    'mean': float(clim[f'{var}_mean'].values),
                    'std': float(clim[f'{var}_std'].values)
                }
    
    def _normalize(self, data: np.ndarray, var: str) -> np.ndarray:
        """Normalize data using climatology"""
        if var in self.norm_stats:
            stats = self.norm_stats[var]
            return (data - stats['mean']) / (stats['std'] + 1e-8)
        return data
    
    def _denormalize(self, data: np.ndarray, var: str) -> np.ndarray:
        """Denormalize data"""
        if var in self.norm_stats:
            stats = self.norm_stats[var]
            return data * stats['std'] + stats['mean']
        return data
    
    def _compute_indices(self):
        """Compute valid time indices for sampling"""
        n_times = len(self.sl_data.time)
        
        # Create sequences of (history_steps + forecast_steps)
        self.valid_indices = []
        for i in range(self.config.history_steps, n_times - self.config.forecast_steps):
            self.valid_indices.append(i)
        
        # Split train/val/test
        n = len(self.valid_indices)
        if self.split == 'train':
            self.valid_indices = self.valid_indices[:int(0.7 * n)]
        elif self.split == 'val':
            self.valid_indices = self.valid_indices[int(0.7 * n):int(0.85 * n)]
        else:  # test
            self.valid_indices = self.valid_indices[int(0.85 * n):]
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        t_idx = self.valid_indices[idx]
        
        # Get time indices
        hist_start = t_idx - self.config.history_steps
        hist_end = t_idx
        fcst_end = t_idx + self.config.forecast_steps
        
        # Load surface data [history_steps, lat, lon, vars]
        sl_data = []
        for var in self.config.surface_vars:
            data = self.sl_data[var].isel(time=slice(t_idx - self.config.history_steps, t_idx))
            data = self._normalize(data.values, var)
            sl_data.append(data)
        surface = np.stack(sl_data, axis=-1)  # [T, lat, lon, vars]
        
        # Load pressure-level data
        pl_data = []
        for var in self.config.atmosphere_vars:
            data = self.pl_data[var].isel(time=slice(t_idx - self.config.history_steps, t_idx))
            data = self._normalize(data.values, var)
            pl_data.append(data)
        atmosphere = np.stack(pl_data, axis=-1)  # [T, lat, lon, level, vars]
        
        # Get target (forecast_steps ahead)
        target_sl = []
        for var in self.config.surface_vars:
            data = self.sl_data[var].isel(time=slice(t_idx, t_idx + self.config.forecast_steps))
            data = self._normalize(data.values, var)
            target_sl.append(data)
        target_surface = np.stack(target_sl, axis=-1)
        
        # Coordinates
        lats = self.sl_data.latitude.values
        lons = self.sl_data.longitude.values
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Temporal encoding
        time = self.sl_data.time.values[t_idx]
        dt = pd.Timestamp(str(time))
        hour = dt.hour
        day_of_year = dt.dayofyear
        
        lat_grid, lon_grid = np.meshgrid(
            self.sl_data.latitude.values, 
            self.sl_data.longitude.values, 
            indexing='ij'
        )
        
        lat_lon = np.stack([
            np.sin(np.deg2rad(lat_grid)), np.cos(np.deg2rad(lat_grid)),
            np.sin(np.deg2rad(lon_grid)), np.cos(np.deg2rad(lon_grid))
        ], axis=-1)
        
        dt = pd.Timestamp(str(self.sl_data.time.values[t_idx]))
        hour = dt.hour
        day_of_year = dt.dayofyear
        
        temporal = np.array([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_year / 365), np.cos(2 * np.pi * day_of_year / 365)
        ])
        
        # Convert to tensors
        return {
            'surface': torch.from_numpy(surface).float(),
            'atmosphere': torch.from_numpy(atmosphere).float(),
            'lat_lon': torch.from_numpy(lat_lon).float(),
            'temporal': torch.from_numpy(temporal).float(),
            'target_surface': torch.from_numpy(target_surface).float(),
        }


class GraphCastDataModule:
    """PyTorch Lightning DataModule for GraphCast"""
    
    def __init__(self, config: GraphCastDataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ERA5Dataset(self.config, 'train')
            self.val_dataset = ERA5Dataset(self.config, 'val')
        
        if stage == 'test' or stage is None:
            self.test_dataset = ERA5Dataset(self.config, 'test')
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )


def create_graphcast_dataloaders(config: GraphCastDataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    train_ds = ERA5Dataset(config, 'train')
    val_ds = ERA5Dataset(config, 'val')
    test_ds = ERA5Dataset(config, 'test')
    
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    config = GraphCastDataConfig(
        era5_path="/data/era5",
        batch_size=2,
        num_workers=0
    )
    
    # This would fail without actual data, but shows the structure
    print("GraphCast data module created successfully")
    print(f"Config: {config}")