"""
FedAIR Client Model Architecture
Implements the local client model with 1D CNN, GRU, and self-attention
Based on: "FedAIR: Federated Air Quality Forecasting via Cross-Client Self-Attention and Spatiotemporal Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for cross-client attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        residual = x
        x = self.layer_norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = output + residual
        
        return output


class FedAIRClient(nn.Module):
    """
    FedAIR Client Model
    Architecture:
    1. 1D CNN for temporal feature extraction
    2. GRU for temporal modeling
    3. Self-attention for cross-client attention
    4. Output head for PM2.5 prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        cnn_filters: list = [32, 64, 128],
        cnn_kernel_size: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        attention_dim: int = 128,
        attention_heads: int = 8,
        dropout: float = 0.2,
        output_dim: int = 24,
        sequence_length: int = 24
    ):
        super(FedAIRClient, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # 1D CNN layers for temporal feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in cnn_filters:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=cnn_kernel_size,
                        padding=cnn_kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # Self-attention layer
        self.attention = SelfAttention(
            embed_dim=gru_hidden,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Output head for prediction
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        
        Returns:
            (batch_size, output_dim) - PM2.5 predictions for future horizon
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN: (batch, channels, sequence)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape back for RNN: (batch, sequence, channels)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, cnn_filters[-1])
        
        # Apply GRU
        gru_out, hidden = self.gru(x)  # gru_out: (batch_size, seq_len, gru_hidden)
        
        # Apply self-attention
        attn_out = self.attention(gru_out)  # (batch_size, seq_len, gru_hidden)
        
        # Use the last time step's output
        last_hidden = attn_out[:, -1, :]  # (batch_size, gru_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Generate predictions
        output = self.output_head(last_hidden)  # (batch_size, output_dim)
        
        return output
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get client embeddings for server aggregation
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        
        Returns:
            (batch_size, gru_hidden) - Client embeddings
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape back for RNN
        x = x.transpose(1, 2)
        
        # Apply GRU
        gru_out, hidden = self.gru(x)
        
        # Apply self-attention
        attn_out = self.attention(gru_out)
        
        # Return last hidden state as embedding
        return attn_out[:, -1, :]  # (batch_size, gru_hidden)


def create_client_model(config: dict) -> FedAIRClient:
    """
    Factory function to create a FedAIR client model
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        FedAIRClient model instance
    """
    return FedAIRClient(
        input_dim=config.get("input_dim", 7),
        cnn_filters=config.get("cnn_filters", [32, 64, 128]),
        cnn_kernel_size=config.get("cnn_kernel_size", 3),
        gru_hidden=config.get("gru_hidden", 128),
        gru_layers=config.get("gru_layers", 2),
        attention_dim=config.get("attention_dim", 128),
        attention_heads=config.get("attention_heads", 8),
        dropout=config.get("dropout", 0.2),
        output_dim=config.get("output_dim", 24),
        sequence_length=config.get("sequence_length", 24)
    )


if __name__ == "__main__":
    # Test the model
    print("=== Testing FedAIR Client Model ===")
    
    from utils.config import CLIENT_CONFIG
    
    model = create_client_model(CLIENT_CONFIG)
    
    # Test forward pass
    batch_size = 32
    seq_len = 24
    input_dim = CLIENT_CONFIG["input_dim"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output = model(x)
        embeddings = model.get_embeddings(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

