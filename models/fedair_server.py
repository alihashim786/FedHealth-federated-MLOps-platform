"""
FedAIR Server Model Architecture
Implements global model aggregation with FedAvg/FedProx and global self-attention fusion
Based on: "FedAIR: Federated Air Quality Forecasting via Cross-Client Self-Attention and Spatiotemporal Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalAttentionFusion(nn.Module):
    """
    Global self-attention fusion layer for aggregating client embeddings
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GlobalAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Multi-layer attention
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, client_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse client embeddings using self-attention
        
        Args:
            client_embeddings: (batch_size, num_clients, embed_dim)
        
        Returns:
            (batch_size, num_clients, embed_dim) - Fused embeddings
        """
        x = client_embeddings
        
        for i, (attn_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Self-attention
            residual = x
            x = layer_norm(x)
            attn_out, _ = attn_layer(x, x, x)
            x = residual + self.dropout(attn_out)
            
            # Feed-forward
            residual = x
            x = layer_norm(x)
            x = residual + self.feed_forward(x)
        
        return x


class FedAIRServer:
    """
    FedAIR Server for model aggregation
    Supports FedAvg and FedProx aggregation methods
    """
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",
        fedprox_mu: float = 0.01,
        attention_dim: int = 128,
        attention_heads: int = 8,
        fusion_layers: int = 2
    ):
        self.aggregation_method = aggregation_method
        self.fedprox_mu = fedprox_mu
        self.global_model = None
        self.global_attention = GlobalAttentionFusion(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            num_layers=fusion_layers
        )
    
    def initialize_global_model(self, client_model: nn.Module):
        """Initialize global model from a client model"""
        self.global_model = copy.deepcopy(client_model)
        logger.info("Initialized global model")
    
    def fedavg_aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: List[float] = None
    ) -> nn.Module:
        """
        Federated Averaging (FedAvg) aggregation
        
        Args:
            client_models: List of client model states
            client_weights: Optional weights for each client (default: uniform)
        
        Returns:
            Aggregated global model
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        # Get parameter names from first model
        first_model_state = client_models[0].state_dict()
        
        for param_name in first_model_state.keys():
            # Weighted average of parameters
            aggregated_param = torch.zeros_like(first_model_state[param_name])
            
            for model, weight in zip(client_models, client_weights):
                aggregated_param += weight * model.state_dict()[param_name]
            
            aggregated_state[param_name] = aggregated_param
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        logger.info(f"Aggregated {len(client_models)} client models using FedAvg")
        
        return self.global_model
    
    def fedprox_aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: List[float] = None,
        global_model: nn.Module = None
    ) -> nn.Module:
        """
        FedProx aggregation with proximal term
        
        Args:
            client_models: List of client model states
            client_weights: Optional weights for each client
            global_model: Previous global model (for proximal term)
        
        Returns:
            Aggregated global model
        """
        if global_model is None:
            global_model = self.global_model
        
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        aggregated_state = {}
        first_model_state = client_models[0].state_dict()
        global_state = global_model.state_dict()
        
        for param_name in first_model_state.keys():
            # Weighted average with proximal term
            aggregated_param = torch.zeros_like(first_model_state[param_name])
            
            for model, weight in zip(client_models, client_weights):
                client_param = model.state_dict()[param_name]
                global_param = global_state[param_name]
                
                # FedProx: weighted average + proximal term
                aggregated_param += weight * (client_param - self.fedprox_mu * (client_param - global_param))
            
            aggregated_state[param_name] = aggregated_param
        
        self.global_model.load_state_dict(aggregated_state)
        
        logger.info(f"Aggregated {len(client_models)} client models using FedProx (mu={self.fedprox_mu})")
        
        return self.global_model
    
    def aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: List[float] = None,
        client_data_sizes: List[int] = None
    ) -> nn.Module:
        """
        Aggregate client models using configured method
        
        Args:
            client_models: List of client model states
            client_weights: Optional explicit weights
            client_data_sizes: Optional data sizes for weighted aggregation
        
        Returns:
            Aggregated global model
        """
        # Use data sizes for weighting if provided
        if client_weights is None and client_data_sizes is not None:
            total_size = sum(client_data_sizes)
            client_weights = [size / total_size for size in client_data_sizes]
        
        if self.aggregation_method == "fedavg":
            return self.fedavg_aggregate(client_models, client_weights)
        elif self.aggregation_method == "fedprox":
            return self.fedprox_aggregate(client_models, client_weights, self.global_model)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def fuse_client_embeddings(
        self,
        client_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse client embeddings using global attention
        
        Args:
            client_embeddings: (batch_size, num_clients, embed_dim)
        
        Returns:
            (batch_size, num_clients, embed_dim) - Fused embeddings
        """
        return self.global_attention(client_embeddings)
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model"""
        return self.global_model


def create_server(config: dict) -> FedAIRServer:
    """
    Factory function to create a FedAIR server
    
    Args:
        config: Server configuration dictionary
    
    Returns:
        FedAIRServer instance
    """
    return FedAIRServer(
        aggregation_method=config.get("aggregation_method", "fedavg"),
        fedprox_mu=config.get("fedprox_mu", 0.01),
        attention_dim=config.get("attention_dim", 128),
        attention_heads=config.get("attention_heads", 8),
        fusion_layers=config.get("fusion_layers", 2)
    )


if __name__ == "__main__":
    # Test server aggregation
    print("=== Testing FedAIR Server ===")
    
    from utils.config import SERVER_CONFIG, CLIENT_CONFIG
    from models.fedair_client import create_client_model
    
    # Create server
    server = create_server(SERVER_CONFIG)
    
    # Create dummy client models
    client_models = [create_client_model(CLIENT_CONFIG) for _ in range(3)]
    
    # Initialize global model
    server.initialize_global_model(client_models[0])
    
    # Test aggregation
    aggregated_model = server.aggregate(
        client_models,
        client_data_sizes=[100, 200, 150]
    )
    
    print(f"Server aggregation method: {server.aggregation_method}")
    print(f"Global model created: {aggregated_model is not None}")
    
    # Test embedding fusion
    batch_size = 4
    num_clients = 3
    embed_dim = 128
    
    client_embeddings = torch.randn(batch_size, num_clients, embed_dim)
    fused_embeddings = server.fuse_client_embeddings(client_embeddings)
    
    print(f"Client embeddings shape: {client_embeddings.shape}")
    print(f"Fused embeddings shape: {fused_embeddings.shape}")

