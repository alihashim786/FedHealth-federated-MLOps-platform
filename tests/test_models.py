"""
Tests for FedAIR model architectures
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.fedair_client import FedAIRClient, create_client_model
from models.fedair_server import FedAIRServer, create_server
from utils.config import CLIENT_CONFIG, SERVER_CONFIG


def test_client_model_creation():
    """Test client model creation"""
    model = create_client_model(CLIENT_CONFIG)
    assert isinstance(model, FedAIRClient)


def test_client_forward():
    """Test client model forward pass"""
    model = create_client_model(CLIENT_CONFIG)
    batch_size = 8
    seq_len = 24
    input_dim = CLIENT_CONFIG["input_dim"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, CLIENT_CONFIG["output_dim"])


def test_client_embeddings():
    """Test client embedding extraction"""
    model = create_client_model(CLIENT_CONFIG)
    batch_size = 8
    seq_len = 24
    input_dim = CLIENT_CONFIG["input_dim"]
    
    x = torch.randn(batch_size, seq_len, input_dim)
    embeddings = model.get_embeddings(x)
    
    assert embeddings.shape == (batch_size, CLIENT_CONFIG["gru_hidden"])


def test_server_creation():
    """Test server creation"""
    server = create_server(SERVER_CONFIG)
    assert isinstance(server, FedAIRServer)


def test_server_aggregation():
    """Test server aggregation"""
    server = create_server(SERVER_CONFIG)
    client_models = [create_client_model(CLIENT_CONFIG) for _ in range(3)]
    
    server.initialize_global_model(client_models[0])
    aggregated = server.aggregate(client_models, client_data_sizes=[100, 200, 150])
    
    assert aggregated is not None

