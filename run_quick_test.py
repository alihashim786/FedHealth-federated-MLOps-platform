"""
Quick test script - verifies system setup without full training
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("FedAIR Quick System Test")
print("=" * 60)

try:
    # 1. Test data loading
    print("\n[1/4] Testing data ingestion...")
    from data_ingestion.node_simulator import NodeSimulator
    simulator = NodeSimulator()
    node_info = simulator.get_node_info()
    print(f"[OK] Loaded {len(node_info)} stations")
    
    # 2. Test preprocessing
    print("\n[2/4] Testing data preprocessing...")
    from data_preprocessing.preprocess import DataPreprocessor
    from data_preprocessing.feature_engineering import FeatureEngineer, SequenceBuilder
    from utils.config import STATIONS, SEQUENCE_LENGTH, PREDICTION_HORIZON
    
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    sequence_builder = SequenceBuilder(SEQUENCE_LENGTH, PREDICTION_HORIZON)
    
    station = STATIONS[0]
    node_data = simulator.get_node_data(station)
    if not node_data.empty:
        processed = preprocessor.preprocess_node(node_data, station)
        train_df = feature_engineer.engineer_features(processed["train"])
        feature_cols = [col for col in train_df.columns if col not in ["pubtime", "PM2.5"]]
        X, y = sequence_builder.create_sequences(train_df, feature_cols)
        print(f"[OK] Preprocessing successful: X shape {X.shape}, y shape {y.shape}")
    
    # 3. Test model
    print("\n[3/4] Testing model architecture...")
    import torch
    from models.fedair_client import create_client_model
    from models.fedair_server import create_server
    from utils.config import CLIENT_CONFIG, SERVER_CONFIG
    
    client_model = create_client_model(CLIENT_CONFIG)
    server = create_server(SERVER_CONFIG)
    server.initialize_global_model(client_model)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, SEQUENCE_LENGTH, CLIENT_CONFIG["input_dim"])
    with torch.no_grad():
        output = client_model(x)
    print(f"[OK] Model test successful: output shape {output.shape}")
    
    # 4. Test federated components
    print("\n[4/4] Testing federated learning components...")
    from fed_learning.aggregator import FederatedAggregator
    from fed_learning.federated_trainer import FederatedTrainer
    
    aggregator = FederatedAggregator(server)
    trainer = FederatedTrainer(server, aggregator, device="cpu")
    print("[OK] Federated components initialized")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed! System is ready.")
    print("=" * 60)
    print("\nTo run full training:")
    print("  python run_training.py")
    print("\nOr:")
    print("  python training/train_federated.py")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

