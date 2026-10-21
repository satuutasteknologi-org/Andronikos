# GraphCast Model Definition (PyTorch)

"""
GraphCast implementation in PyTorch for training, compatible with Burn inference.

Architecture based on DeepMind's GraphCast:
- Encoder: Node/Edge embeddings
- Processor: 12 GNN layers with message passing
- Temporal: 3 attention layers
- Decoder: Multi-task heads for 17 variables

Reference: "GraphCast: Learning skillful medium-range global weather forecasting" (Lam et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GraphCastConfig:
    """GraphCast model configuration"""
    hidden_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    surface_vars: int = 5
    atmosphere_vars: int = 6
    pressure_levels: int = 13
    dropout: float = 0.1
    num_mesh_nodes: int = 38927200  # Indonesia 0.5km grid
    edge_dim: int = 4  # relative position encoding
    temporal_dim: int = 4  # sin/cos hour + sin/cos day_of_year


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PerformerAttention(nn.Module):
    """Performer attention (O(N) linear attention)"""
    
    def __init__(self, dim: int, num_heads: int = 8, num_features: int = 256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_features = num_features
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
        # Random features for Performer
        self.register_buffer(
            'projection_matrix',
            torch.randn(num_features, self.head_dim)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Performer: random feature map
        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)
        
        # Linear attention
        kv = torch.einsum('bhnd,bhnf->bhdf', k_prime, v)
        z = torch.einsum('bhnd,bhd->bhn', q_prime, k_prime.sum(dim=2)).clamp(min=1e-6)
        out = torch.einsum('bhdf,bhn->bhnd', kv, q_prime) / z.unsqueeze(-1)
        
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.dropout(self.proj(out))
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Random feature map for Performer"""
        proj = torch.einsum('bhnd,fd->bhnf', x, self.projection_matrix)
        return F.relu(proj) / math.sqrt(self.num_features)


class GNNLayer(MessagePassing):
    """Graph Neural Network Layer with edge and node updates"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        # x: [N, hidden_dim], edge_index: [2, E]
        
        # Edge update
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col]], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        edge_out = self.edge_mlp(edge_input)
        
        # Message passing
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_out)
        
        # Node update with residual
        node_input = torch.cat([x, aggr_out], dim=-1)
        node_out = self.node_mlp(node_input)
        
        # Residual + norm
        x = self.norm1(x + node_out)
        
        # Second residual block
        residual = x
        x = self.norm2(x + self.node_mlp(torch.cat([x, x], dim=-1)))
        
        return x + residual
    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return edge_attr


class TemporalAttention(nn.Module):
    """Temporal self-attention for sequence modeling"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, N, hidden_dim] - temporal sequence"""
        # Self-attention over time dimension
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        return self.norm(x)


class GraphCastEncoder(nn.Module):
    """Input encoder for surface, atmosphere, and positional features"""
    
    def __init__(self, config: GraphCastConfig):
        super().__init__()
        self.config = config
        
        # Surface encoder: 5 vars * 4 time steps = 20 features
        self.surface_encoder = nn.Linear(config.surface_vars * 4, config.hidden_dim)
        
        # Atmosphere encoder: 6 vars * 13 levels * 4 time steps = 312 features
        atm_features = config.atmosphere_vars * config.pressure_levels * 4
        self.atmosphere_encoder = nn.Linear(atm_features, config.hidden_dim)
        
        # Positional encoding (lat/lon)
        self.lat_lon_encoder = nn.Linear(4, config.hidden_dim)  # sin/cos lat, sin/cos lon
        
        # Temporal encoding
        self.temporal_encoder = nn.Linear(4, config.hidden_dim)  # sin/cos hour, sin/cos day
        
        # Input projection
        self.input_proj = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
    
    def forward(self, surface: torch.Tensor, atmosphere: torch.Tensor,
                lat_lon: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surface: [N, surface_vars * 4]
            atmosphere: [N, atmosphere_vars * pressure_levels * 4]
            lat_lon: [N, 4] (sin/cos lat, sin/cos lon)
            temporal: [N, 4] (sin/cos hour, sin/cos day)
        Returns:
            [N, hidden_dim]
        """
        surface_enc = self.surface_encoder(surface)
        atm_enc = self.atmosphere_encoder(atmosphere)
        lat_lon_enc = self.lat_lon_encoder(lat_lon)
        temp_enc = self.temporal_encoder(temporal)
        
        combined = torch.cat([surface_enc, atm_enc, lat_lon_enc, temp_enc], dim=-1)
        return self.input_proj(combined)


class GraphCastProcessor(nn.Module):
    """GraphCast processor: GNN layers + temporal attention"""
    
    def __init__(self, config: GraphCastConfig):
        super().__init__()
        self.config = config
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Temporal attention layers
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(3)
        ])
        
        self.decoder = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] or [T, N, hidden_dim]
            edge_index: [2, E]
        """
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
        
        # Temporal attention (if temporal dimension exists)
        if x.dim() == 3:  # [T, N, D]
            for attn in self.temporal_attention:
                x = attn(x)
        
        # Decoder
        return self.decoder(x)


class GraphCastDecoder(nn.Module):
    """Multi-task decoder heads for each output variable"""
    
    def __init__(self, config: GraphCastConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Output variables (17 surface variables)
        self.output_vars = [
            'temperature', 'humidity', 'precipitation', 'wind_speed', 'wind_dir',
            'pressure', 'cloud_cover', 'wind_u', 'wind_v', 'dew_point',
            'visibility', 'solar_radiation', 'uv_index', 'cape',
            'soil_moisture', 'soil_temp', 'snow'
        ]
        
        # Each head outputs 4 quantiles (p10, p50, p90, prob_rain for precip)
        self.heads = nn.ModuleDict()
        for var in self.output_vars:
            out_dim = 4 if var == 'precipitation' else 1
            self.heads[var] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, out_dim)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        for var, head in self.heads.items():
            outputs[var] = head(x)
        return outputs


class GraphCast(nn.Module):
    """Complete GraphCast model"""
    
    def __init__(self, config: GraphCastConfig):
        super().__init__()
        self.config = config
        
        self.encoder = GraphCastEncoder(config)
        self.processor = GraphCastProcessor(config)
        self.decoder = GraphCastDecoder(config)
    
    def forward(self, surface: torch.Tensor, atmosphere: torch.Tensor,
                lat_lon: torch.Tensor, temporal: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            surface: [N, surface_vars * 4]
            atmosphere: [N, atmosphere_vars * pressure_levels * 4]
            lat_lon: [N, 4]
            temporal: [N, 4]
            edge_index: [2, E]
            edge_attr: [E, 4] optional
        """
        # Encode
        x = self.encoder(surface, atmosphere, lat_lon, temporal)
        
        # Process
        x = self.processor(x, edge_index)
        
        # Decode
        outputs = self.decoder(x)
        
        return outputs
    
    def load_pretrained_burn(self, checkpoint_path: str):
        """Load checkpoint from Burn format"""
        # This would load .safetensors or Burn checkpoint
        # Implementation depends on exact checkpoint format
        pass
    
    def export_to_burn(self, output_path: str):
        """Export model for Burn inference"""
        # Convert PyTorch state dict to Burn-compatible format
        # Save as .safetensors or Burn's native format
        pass
    
    def export_to_onnx(self, output_path: str, dummy_input: Tuple):
        """Export model to ONNX format"""
        torch.onnx.export(
            self, dummy_input, output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['surface', 'atmosphere', 'lat_lon', 'temporal', 'edge_index'],
            output_names=[f'output_{var}' for var in self.decoder.output_vars],
            dynamic_axes={
                'surface': {0: 'batch'},
                'atmosphere': {0: 'batch'},
                'lat_lon': {0: 'batch'},
                'temporal': {0: 'batch'},
                'edge_index': {1: 'edges'}
            }
        )


def create_graphcast(config: Optional[GraphCastConfig] = None) -> GraphCast:
    """Factory function to create GraphCast model"""
    if config is None:
        config = GraphCastConfig()
    return GraphCast(config)


if __name__ == "__main__":
    # Test model creation
    config = GraphCastConfig()
    model = create_graphcast(config)
    print(f"GraphCast model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    N = 100
    config = GraphCastConfig()
    model = create_graphcast(config)
    
    surface = torch.randn(N, config.surface_vars * 4)
    atmosphere = torch.randn(N, config.atmosphere_vars * config.pressure_levels * 4)
    lat_lon = torch.randn(N, 4)
    temporal = torch.randn(N, 4)
    edge_index = torch.randint(0, N, (2, N * 4))
    
    outputs = model(surface, atmosphere, lat_lon, temporal, edge_index)
    print("Output keys:", list(outputs.keys()))
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")