"""
Configuration file for FedAIR Federated Learning System
Centralized configuration for all components
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
# Dataset location: override with FEDAIR_DATA_ROOT env var, defaults to <project>/data/pollutant
DATA_ROOT = Path(os.environ.get("FEDAIR_DATA_ROOT", PROJECT_ROOT / "data" / "pollutant"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Dataset configuration
STATIONS = ["CCM", "MGQ", "OTZX", "PK", "RJL", "SXL", "XLDXC", "XWH", "ZHM"]
NUM_STATIONS = len(STATIONS)
TARGET_COLUMN = "PM2.5"
FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2", "AQI"]
TIMESTAMP_COLUMN = "pubtime"

# Data preprocessing
LAG_FEATURES = [1, 3, 6, 12, 24]  # hours
SEQUENCE_LENGTH = 24  # 24 hours of historical data
PREDICTION_HORIZON = 24  # Predict 24 hours ahead
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model architecture (FedAIR)
# Input dim: original features + temporal features (11) + lag features (35) = 7 + 11 + 35 = 53
# But we exclude timestamp and target, so actual feature count after engineering
CLIENT_CONFIG = {
    "input_dim": 52,  # Will be set dynamically based on actual feature count
    "cnn_filters": [32, 64, 128],
    "cnn_kernel_size": 3,
    "gru_hidden": 128,
    "gru_layers": 2,
    "attention_dim": 128,
    "attention_heads": 8,
    "dropout": 0.2,
    "output_dim": PREDICTION_HORIZON
}

SERVER_CONFIG = {
    "aggregation_method": "fedavg",  # or "fedprox"
    "fedprox_mu": 0.01,
    "attention_dim": 128,
    "attention_heads": 8,
    "fusion_layers": 2
}

# Federated learning
FEDERATED_CONFIG = {
    "num_rounds": 5,  # Reduced for quick demo - change to 100 for full training
    "num_clients_per_round": 5,  # Select 5 out of 9 clients per round
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "secure_aggregation": True,
    "differential_privacy": False,
    "dp_epsilon": 1.0,
    "dp_delta": 1e-5
}

# Training
TRAINING_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "num_workers": 0,  # Set to 0 for Windows compatibility
    "pin_memory": False,  # Disable for CPU
    "save_checkpoints": True,
    "checkpoint_freq": 10,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.001
}

# Drift detection
DRIFT_CONFIG = {
    "detection_method": "kl_divergence",  # or "adwin"
    "kl_threshold": 0.1,
    "adwin_delta": 0.002,
    "window_size": 1000,
    "check_frequency": 100,  # Check every 100 samples
    "retrain_threshold": 0.15
}

# MLOps
MLFLOW_CONFIG = {
    "tracking_uri": "http://localhost:5000",
    "experiment_name": "fedair_air_quality",
    "log_artifacts": True,
    "log_models": True
}

# Monitoring
MONITORING_CONFIG = {
    "prometheus_port": 9090,
    "api_port": 8000,
    "update_frequency": 60,  # seconds
    "alert_threshold_pm25": 150,  # µg/m³
    "latency_threshold_ms": 100
}

# Dashboard
DASHBOARD_CONFIG = {
    "port": 8501,
    "host": "localhost",
    "update_interval": 300  # 5 minutes
}

# Docker & Kubernetes
DOCKER_CONFIG = {
    "image_name": "fedair-server",
    "tag": "latest",
    "port": 8000
}

K8S_CONFIG = {
    "namespace": "fedair",
    "replicas": 3,
    "cpu_request": "500m",
    "cpu_limit": "2000m",
    "memory_request": "1Gi",
    "memory_limit": "4Gi"
}

