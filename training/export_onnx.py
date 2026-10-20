# ONNX Export Script

"""
Export trained PyTorch models to ONNX format for Rust inference.
"""

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from graphcast.model import GraphCast, GraphCastConfig
from downscaler.model import DownscalerNet, DownscalerConfig

logger = logging.getLogger(__name__)


def export_graphcast_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: GraphCastConfig,
    dummy_input: Tuple[torch.Tensor, ...]
) -> None:
    """Export GraphCast model to ONNX"""
    
    logger.info(f"Loading GraphCast checkpoint from {checkpoint_path}")
    model = GraphCast(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Export to ONNX
    logger.info(f"Exporting GraphCast to ONNX: {output_path}")
    
    input_names = ['surface', 'atmosphere', 'lat_lon', 'temporal', 'edge_index']
    output_names = [f'output_{var}' for var in [
        'temperature', 'humidity', 'precipitation', 'wind_speed', 'wind_dir',
        'pressure', 'cloud_cover', 'wind_u', 'wind_v', 'dew_point',
        'visibility', 'solar_radiation', 'uv_index', 'cape',
        'soil_moisture', 'soil_temp', 'snow'
    ]]
    
    dynamic_axes = {
        'surface': {0: 'batch', 1: 'nodes'},
        'atmosphere': {0: 'batch', 1: 'nodes'},
        'lat_lon': {0: 'batch', 1: 'nodes'},
        'temporal': {0: 'batch', 1: 'nodes'},
        'edge_index': {1: 'edges'}
    }
    
    for name in output_names:
        dynamic_axes[name] = {0: 'batch', 1: 'nodes'}
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    logger.info(f"GraphCast exported successfully to {output_path}")


def export_downscaler_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: DownscalerConfig,
    dummy_input: torch.Tensor
) -> None:
    """Export Downscaler model to ONNX"""
    
    logger.info(f"Loading Downscaler checkpoint from {checkpoint_path}")
    model = DownscalerNet(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    logger.info(f"Exporting Downscaler to ONNX: {output_path}")
    
    input_names = ['features']
    output_names = ['output']
    
    dynamic_axes = {
        'features': {0: 'batch'},
        'output': {0: 'batch'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    logger.info(f"Downscaler exported successfully to {output_path}")


def validate_onnx_model(model_path: str) -> bool:
    """Validate ONNX model using onnxruntime"""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure is valid")
        
        # Test inference
        session = ort.InferenceSession(model_path)
        logger.info(f"ONNX model inference session created successfully")
        logger.info(f"Input names: {[i.name for i in session.get_inputs()]}")
        logger.info(f"Output names: {[o.name for o in session.get_outputs()]}")
        
        return True
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def create_dummy_inputs(
    graphcast_config: GraphCastConfig,
    downscaler_config: DownscalerConfig
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """Create dummy inputs for export"""
    
    # GraphCast dummy input
    N = 100  # number of nodes
    graphcast_input = (
        torch.randn(1, graphcast_config.surface_vars * 4),      # surface
        torch.randn(1, graphcast_config.atmosphere_vars * graphcast_config.pressure_levels * 4),  # atmosphere
        torch.randn(1, 4),                                       # lat_lon
        torch.randn(1, 4),                                       # temporal
        torch.randint(0, N, (2, N * 4))                          # edge_index
    )
    
    # Downscaler dummy input
    downscaler_input = torch.randn(1, 14)
    
    return graphcast_input, downscaler_input


@hydra.main(config_path="../configs", config_name="export", version_base="1.3")
def main(config: DictConfig):
    logger.info("Starting ONNX export")
    
    # Load configs
    graphcast_config = GraphCastConfig(
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        surface_vars=config.model.surface_vars,
        atmosphere_vars=config.model.atmosphere_vars,
        pressure_levels=config.model.pressure_levels,
        dropout=config.model.dropout
    )
    
    downscaler_config = DownscalerConfig(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout
    )
    
    # Create dummy inputs
    graphcast_input, downscaler_input = create_dummy_inputs(
        graphcast_config, downscaler_config
    )
    
    # Export GraphCast
    if config.export.graphcast:
        export_graphcast_to_onnx(
            checkpoint_path=config.export.graphcast_checkpoint,
            output_path=config.export.graphcast_output,
            config=graphcast_config,
            dummy_input=graphcast_input
        )
        
        # Validate
        validate_onnx_model(config.export.graphcast_output)
    
    # Export Downscaler
    if config.export.downscaler:
        export_downscaler_to_onnx(
            checkpoint_path=config.export.downscaler_checkpoint,
            output_path=config.export.downscaler_output,
            config=DownscalerConfig(),
            dummy_input=torch.randn(1, 14)
        )
        
        # Validate
        validate_onnx_model(config.export.downscaler_output)
    
    logger.info("ONNX export completed successfully!")


if __name__ == "__main__":
    main()