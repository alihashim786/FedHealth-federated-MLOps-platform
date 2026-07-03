"""
Quick runner script for FedAIR training
Can be executed with: python run_training.py
Or with full path: C:\path\to\python.exe run_training.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("FedAIR Training Pipeline")
print("=" * 60)

try:
    # Test imports
    print("\n[1/5] Testing imports...")
    from data_ingestion.node_simulator import NodeSimulator
    from data_preprocessing.preprocess import DataPreprocessor
    from data_preprocessing.feature_engineering import FeatureEngineer
    print("[OK] All imports successful")
    
    # Test data loading
    print("\n[2/5] Loading data...")
    simulator = NodeSimulator()
    node_info = simulator.get_node_info()
    print(f"[OK] Loaded {len(node_info)} stations")
    for station in list(node_info.keys())[:3]:
        print(f"  - {station}: {node_info[station]['num_records']} records")
    
    # Test preprocessing
    print("\n[3/5] Testing preprocessing...")
    preprocessor = DataPreprocessor()
    station = list(node_info.keys())[0]
    node_data = simulator.get_node_data(station)
    if not node_data.empty:
        processed = preprocessor.preprocess_node(node_data, station, split=False)
        print(f"[OK] Preprocessing successful for {station}")
    
    # Test model creation
    print("\n[4/5] Testing model creation...")
    from models.fedair_client import create_client_model
    from utils.config import CLIENT_CONFIG
    model = create_client_model(CLIENT_CONFIG)
    print(f"[OK] Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run training
    print("\n[5/5] Starting federated training...")
    print("Note: This will take some time. For a quick test, reduce num_rounds in config.py")
    print("\n" + "=" * 60)
    
    from training.train_federated import main
    main()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n[ERROR] Import error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

