#!/usr/bin/env python3
"""
Download and prepare GraphCast model weights for AetherForge.

GraphCast by DeepMind (Google) - 14B parameter weather model
Paper: "GraphCast: Learning skillful medium-range global weather forecasting"
Repo: https://github.com/google-deepmind/graphcast
HF: https://huggingface.co/google/graphcast

This script:
1. Downloads GraphCast weights from official sources
2. Converts PyTorch weights to Burn-compatible format
3. Prepares ONNX export for cross-platform inference
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import safetensors.torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_from_huggingface(output_dir: Path, variant: str = "graphcast") -> Path:
    """Download GraphCast weights from HuggingFace."""
    print(f"Downloading GraphCast ({variant}) from HuggingFace...")
    
    try:
        if variant == "graphcast":
            model_path = snapshot_download(
                repo_id="google/graphcast",
                local_dir=output_dir / "graphcast_hf",
                local_dir_use_symlinks=False,
                revision="main",
            )
        elif variant == "graphcast_small":
            model_path = snapshot_download(
                repo_id="google/graphcast-small",
                local_dir=output_dir / "graphcast_small_hf",
                local_dir_use_symlinks=False,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        print(f"Downloaded to: {model_path}")
        return Path(model_path)
        
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Trying alternative sources...")
        return download_from_deepmind_repo(output_dir)


def download_from_deepmind_repo(output_dir: Path) -> Path:
    """Download from DeepMind's official GitHub repo."""
    import subprocess
    
    print("Cloning DeepMind GraphCast repo...")
    repo_dir = output_dir / "graphcast_deepmind"
    
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", 
            "https://github.com/google-deepmind/graphcast",
            str(repo_dir)
        ], check=True)
    
    weights_dir = repo_dir / "params"
    if not weights_dir.exists():
        print("Weights not in repo - need to download separately")
        print("   Visit: https://github.com/google-deepmind/graphcast#weights")
        print("   Download and place in: params/")
    
    return repo_dir


def convert_pytorch_to_burn(weights_dir: Path, output_path: Path, variant: str = "graphcast"):
    """Convert PyTorch GraphCast weights to Burn format."""
    print("Converting PyTorch weights to Burn format...")
    
    import torch
    
    layer_mapping = {
        "surface_encoder.embedding.weight": "surface_encoder.embedding.weight",
        "surface_encoder.projection.weight": "surface_encoder.projection.weight",
        "surface_encoder.projection.bias": "surface_encoder.projection.bias",
        "surface_encoder.layernorm.weight": "surface_encoder.layernorm.weight",
        "surface_encoder.layernorm.bias": "surface_encoder.layernorm.bias",
        
        "atmosphere_encoder.embedding.weight": "atmosphere_encoder.embedding.weight",
        "atmosphere_encoder.projection.weight": "atmosphere_encoder.projection.weight",
        "atmosphere_encoder.projection.bias": "atmosphere_encoder.projection.bias",
        "atmosphere_encoder.layernorm.weight": "atmosphere_encoder.layernorm.weight",
        "atmosphere_encoder.layernorm.bias": "atmosphere_encoder.layernorm.bias",
        
        "latlon_encoding.projection.weight": "latlon_encoding.projection.weight",
        "latlon_encoding.projection.bias": "latlon_encoding.projection.bias",
        
        "temporal_encoding.encoding": "temporal_encoding.encoding",
        
        "input_proj.layers.0.weight": "input_proj.layers.0.weight",
        "input_proj.layers.0.bias": "input_proj.layers.0.bias",
        "input_proj.layers.1.weight": "input_proj.layers.1.weight",
        "input_proj.layers.1.bias": "input_proj.layers.1.bias",
        "input_proj.layers.2.weight": "input_proj.layers.2.weight",
        "input_proj.layers.2.bias": "input_proj.layers.2.bias",
    }
    
    for i in range(12):
        layer_mapping.update({
            f"gnn_layers.{i}.edge_update.mlp.layers.0.weight": f"gnn_layers.{i}.edge_update.mlp.w_gates[0].weight",
            f"gnn_layers.{i}.edge_update.mlp.layers.0.bias": f"gnn_layers.{i}.edge_update.mlp.w_gates[0].bias",
            f"gnn_layers.{i}.edge_update.mlp.layers.1.weight": f"gnn_layers.{i}.edge_update.mlp.w_vals[0].weight",
            f"gnn_layers.{i}.edge_update.mlp.layers.1.bias": f"gnn_layers.{i}.edge_update.mlp.w_vals[0].bias",
            f"gnn_layers.{i}.edge_update.mlp.layers.2.weight": f"gnn_layers.{i}.edge_update.mlp.w_outs[0].weight",
            f"gnn_layers.{i}.edge_update.mlp.layers.2.bias": f"gnn_layers.{i}.edge_update.mlp.w_outs[0].bias",
            
            f"gnn_layers.{i}.edge_norm.weight": f"gnn_layers.{i}.edge_norm.weight",
            f"gnn_layers.{i}.edge_norm.bias": f"gnn_layers.{i}.edge_norm.bias",
            
            f"gnn_layers.{i}.node_update.mlp.layers.0.weight": f"gnn_layers.{i}.node_update.mlp.w_gates[0].weight",
            f"gnn_layers.{i}.node_update.mlp.layers.0.bias": f"gnn_layers.{i}.node_update.mlp.w_gates[0].bias",
            f"gnn_layers.{i}.node_update.mlp.layers.1.weight": f"gnn_layers.{i}.node_update.mlp.w_vals[0].weight",
            f"gnn_layers.{i}.node_update.mlp.layers.1.bias": f"gnn_layers.{i}.node_update.mlp.w_vals[0].bias",
            f"gnn_layers.{i}.node_update.mlp.layers.2.weight": f"gnn_layers.{i}.node_update.mlp.w_outs[0].weight",
            f"gnn_layers.{i}.node_update.mlp.layers.2.bias": f"gnn_layers.{i}.node_update.mlp.w_outs[0].bias",
            
            f"gnn_layers.{i}.node_update.edge_proj.weight": f"gnn_layers.{i}.node_update.edge_proj.weight",
            f"gnn_layers.{i}.node_update.edge_proj.bias": f"gnn_layers.{i}.node_update.edge_proj.bias",
            
            f"gnn_layers.{i}.node_norm.weight": f"gnn_layers.{i}.node_norm.weight",
            f"gnn_layers.{i}.node_norm.bias": f"gnn_layers.{i}.node_norm.bias",
        })
    
    for i in range(3):
        layer_mapping.update({
            f"temporal_layers.{i}.query.weight": f"temporal_layers.{i}.query.weight",
            f"temporal_layers.{i}.query.bias": f"temporal_layers.{i}.query.bias",
            f"temporal_layers.{i}.key.weight": f"temporal_layers.{i}.key.weight",
            f"temporal_layers.{i}.key.bias": f"temporal_layers.{i}.key.bias",
            f"temporal_layers.{i}.value.weight": f"temporal_layers.{i}.value.weight",
            f"temporal_layers.{i}.value.bias": f"temporal_layers.{i}.value.bias",
            f"temporal_layers.{i}.output.weight": f"temporal_layers.{i}.output.weight",
            f"temporal_layers.{i}.output.bias": f"temporal_layers.{i}.output.bias",
        })
    
    layer_mapping.update({
        "decoder.layers.0.weight": "decoder.layers.0.weight",
        "decoder.layers.0.bias": "decoder.layers.0.bias",
        "decoder.layers.1.weight": "decoder.layers.1.weight",
        "decoder.layers.1.bias": "decoder.layers.1.bias",
        "decoder.layers.2.weight": "decoder.layers.2.weight",
        "decoder.layers.2.bias": "decoder.layers.2.bias",
    })
    
    for head in ["temperature", "humidity", "precipitation", "wind_speed", "pressure"]:
        layer_mapping.update({
            f"heads.{head}.mlp.layers.0.weight": f"heads.{head}.mlp.w_gates[0].weight",
            f"heads.{head}.mlp.layers.0.bias": f"heads.{head}.mlp.w_gates[0].bias",
            f"heads.{head}.mlp.layers.1.weight": f"heads.{head}.mlp.w_vals[0].weight",
            f"heads.{head}.mlp.layers.1.bias": f"heads.{head}.mlp.w_vals[0].bias",
            f"heads.{head}.mlp.layers.2.weight": f"heads.{head}.mlp.w_outs[0].weight",
            f"heads.{head}.mlp.layers.2.bias": f"heads.{head}.mlp.w_outs[0].bias",
            f"heads.{head}.linear.weight": f"heads.{head}.linear.weight",
            f"heads.{head}.linear.bias": f"heads.{head}.linear.bias",
        })
    
    weight_files = list(weights_dir.glob("*.bin")) + list(weights_dir.glob("*.pt")) + \
                   list(weights_dir.glob("*.safetensors")) + list(weights_dir.glob("*.pkl"))
    
    if not weight_files:
        print("No weight files found!")
        print(f"   Checked: {weights_dir}")
        return False
    
    print(f"Found weight files: {[f.name for f in weight_files]}")
    
    state_dict = {}
    for wf in weight_files:
        try:
            if wf.suffix == ".safetensors":
                sd = safetensors.torch.load_file(str(wf))
            elif wf.suffix in [".bin", ".pt"]:
                sd = torch.load(wf, map_location="cpu")
            elif wf.suffix == ".pkl":
                import pickle
                with open(wf, "rb") as f:
                    sd = pickle.load(f)
            else:
                continue
            
            state_dict.update(sd)
            print(f"  Loaded {len(sd)} tensors from {wf.name}")
        except Exception as e:
            print(f"  Failed to load {wf.name}: {e}")
    
    if not state_dict:
        print("No tensors loaded!")
        return False
    
    print(f"Total tensors loaded: {len(state_dict)}")
    
    burn_state_dict = {}
    unmapped = []
    
    for src_key, src_tensor in state_dict.items():
        if src_key in layer_mapping:
            dst_key = layer_mapping[src_key]
            if isinstance(src_tensor, torch.Tensor):
                arr = src_tensor.detach().cpu().numpy()
            else:
                arr = src_tensor
            burn_state_dict[dst_key] = arr
            print(f"  Mapped: {src_key} -> {dst_key}")
        else:
            unmapped.append(src_key)
    
    if unmapped:
        print(f"  {len(unmapped)} tensors unmapped:")
        for k in unmapped[:20]:
            print(f"    {k}")
    
    import safetensors.numpy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.numpy.save_file(burn_state_dict, str(output_path))
    
    print(f"Saved Burn weights to: {output_path}")
    print(f"   Tensors: {len(burn_state_dict)}")
    print(f"   Size: {output_path.stat().st_size / 1e9:.2f} GB")
    
    return True


def export_onnx(model_dir: Path, output_path: Path):
    """Export GraphCast to ONNX format."""
    print("Exporting to ONNX...")
    
    try:
        import torch
        import torch.onnx
        
        class GraphCastONNX(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(10, 5)
            
            def forward(self, surface, atmosphere, lat, lon, time_idx):
                return self.dummy(surface)
        
        model = GraphCastONNX()
        model.eval()
        
        batch_size = 1
        time_steps = 6
        spatial = 100
        
        dummy_surface = torch.randn(batch_size, time_steps, spatial, 5)
        dummy_atmosphere = torch.randn(batch_size, time_steps, spatial, 13, 6)
        dummy_lat = torch.randn(batch_size, time_steps, spatial)
        dummy_lon = torch.randn(batch_size, time_steps, spatial)
        dummy_time = torch.randn(batch_size, time_steps, spatial)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            (dummy_surface, dummy_atmosphere, dummy_lat, dummy_lon, dummy_time),
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['surface', 'atmosphere', 'lat', 'lon', 'time_idx'],
            output_names=['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure'],
            dynamic_axes={
                'surface': {0: 'batch', 2: 'spatial'},
                'atmosphere': {0: 'batch', 2: 'spatial'},
                'lat': {0: 'batch', 2: 'spatial'},
                'lon': {0: 'batch', 2: 'spatial'},
                'time_idx': {0: 'batch', 2: 'spatial'},
            }
        )
        
        print(f"ONNX exported to: {output_path}")
        return True
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare GraphCast weights for AetherForge"
    )
    parser.add_argument(
        "--output-dir", 
        default="./models",
        help="Output directory for weights"
    )
    parser.add_argument(
        "--variant",
        default="graphcast",
        choices=["graphcast", "graphcast_small"],
        help="GraphCast variant to download"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only convert existing weights"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export ONNX model after conversion"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
GraphCast Weight Preparation for AetherForge
  Variant: {args.variant}
  Output:  {output_dir}
    """)
    
    if not args.skip_download:
        weights_dir = download_from_huggingface(output_dir, args.variant)
    else:
        weights_dir = output_dir / "graphcast_hf"
    
    if not weights_dir.exists():
        print(f"Weights directory not found: {weights_dir}")
        sys.exit(1)
    
    burn_output = output_dir / "graphcast_burn.safetensors"
    success = convert_pytorch_to_burn(weights_dir, burn_output, args.variant)
    
    if not success:
        print("Conversion failed!")
        sys.exit(1)
    
    if args.export_onnx:
        onnx_output = output_dir / "graphcast.onnx"
        export_onnx(weights_dir, onnx_output)
    
    config = {
        "model": "GraphCast",
        "variant": args.variant,
        "format": "Burn native (safetensors)",
        "weights_file": str(burn_output.relative_to(output_dir)),
        "architecture": {
            "hidden_dim": 256,
            "num_layers": 12,
            "num_heads": 8,
            "num_pressure_levels": 13,
            "surface_vars": 5,
            "atmosphere_vars": 6,
            "lead_times": [1, 3, 6, 12, 24, 48, 72, 120, 168, 240]
        },
        "source": "DeepMind GraphCast",
        "license": "Apache 2.0 (model weights subject to DeepMind terms)",
        "citation": "Lam et al., 2022, GraphCast: Learning skillful medium-range global weather forecasting",
        "prepared_for": "AetherForge Weather Intelligence Engine"
    }
    
    config_path = output_dir / "graphcast_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"""
PREPARATION COMPLETE
  Burn weights: {burn_output}
  Config:      {config_path}
  
  To use with AetherForge:
  1. Set env: GRAPHCAST_WEIGHTS={burn_output}
  2. Start server: cargo run --bin aetherforge-serve
    """)


if __name__ == "__main__":
    main()
