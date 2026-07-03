"""
End-to-End Federated Training Pipeline
Orchestrates: ingest → preprocess → federated rounds → evaluation
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from data_ingestion.node_simulator import NodeSimulator
from data_preprocessing.preprocess import DataPreprocessor
from data_preprocessing.feature_engineering import FeatureEngineer, SequenceBuilder
from models.fedair_client import create_client_model
from models.fedair_server import create_server
from fed_learning.aggregator import FederatedAggregator
from fed_learning.federated_trainer import FederatedTrainer
from utils.config import (
    STATIONS, CLIENT_CONFIG, SERVER_CONFIG, FEDERATED_CONFIG,
    TRAINING_CONFIG, MODEL_DIR, OUTPUT_DIR, SEQUENCE_LENGTH, PREDICTION_HORIZON
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedTrainingPipeline:
    """
    Complete federated training pipeline
    """
    
    def __init__(self):
        self.node_simulator = NodeSimulator()
        self.preprocessor = DataPreprocessor(normalization_method="standard")
        self.feature_engineer = FeatureEngineer()
        self.sequence_builder = SequenceBuilder(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON
        )
        
        # Initialize models (will update input_dim after feature engineering)
        self.server = create_server(SERVER_CONFIG)
        self.aggregator = FederatedAggregator(self.server)
        
        # Initialize global model with placeholder (will be updated)
        dummy_config = CLIENT_CONFIG.copy()
        dummy_config["input_dim"] = 52  # Actual feature count after engineering
        dummy_model = create_client_model(dummy_config)
        self.server.initialize_global_model(dummy_model)
        
        self.trainer = FederatedTrainer(
            self.server,
            self.aggregator,
            device=TRAINING_CONFIG["device"]
        )
        
        self.processed_data = {}
        self.training_history = []
    
    def prepare_data(self):
        """Prepare data for all nodes"""
        logger.info("Preparing data for all nodes...")
        
        for station in STATIONS:
            logger.info(f"Processing station: {station}")
            
            # Load node data
            node_data = self.node_simulator.get_node_data(station)
            
            if node_data.empty:
                logger.warning(f"No data for station {station}")
                continue
            
            # Preprocess
            processed = self.preprocessor.preprocess_node(node_data, station)
            
            # Feature engineering
            for split in ["train", "val", "test"]:
                if split in processed:
                    processed[split] = self.feature_engineer.engineer_features(
                        processed[split],
                        include_temporal=True,
                        include_lags=True,
                        include_rolling=False
                    )
            
            # Create sequences
            feature_cols = [
                col for col in processed["train"].columns
                if col not in ["pubtime", "PM2.5"]
            ]
            
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                self.sequence_builder.create_sequences_from_dataframes(
                    processed["train"],
                    processed["val"],
                    processed["test"],
                    feature_cols,
                    target_col="PM2.5"
                )
            
            self.processed_data[station] = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
                "feature_cols": feature_cols
            }
            
            logger.info(
                f"Station {station}: Train={X_train.shape}, "
                f"Val={X_val.shape}, Test={X_test.shape}"
            )
    
    def train(self, num_rounds: int = None):
        """Run federated training"""
        if num_rounds is None:
            num_rounds = FEDERATED_CONFIG["num_rounds"]
        
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        # Prepare training data
        train_data = {
            station: data["train"]
            for station, data in self.processed_data.items()
        }
        
        # Prepare validation data
        val_data = {
            station: data["val"]
            for station, data in self.processed_data.items()
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for round_num in range(1, num_rounds + 1):
            # Federated round
            round_result = self.trainer.federated_round(
                train_data,
                round_num=round_num,
                local_epochs=FEDERATED_CONFIG["local_epochs"]
            )
            
            # Periodic evaluation
            if round_num % 10 == 0 or round_num == num_rounds:
                val_metrics = self.trainer.evaluate_global_model(val_data)
                val_loss = val_metrics["overall"]["rmse"]
                
                logger.info(
                    f"Round {round_num}: Val RMSE = {val_loss:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss - TRAINING_CONFIG["early_stopping_min_delta"]:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.trainer.save_model(MODEL_DIR, round_num=round_num)
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                    logger.info(f"Early stopping at round {round_num}")
                    break
            
            self.training_history.append(round_result)
        
        logger.info("Federated training completed")
    
    def evaluate(self):
        """Evaluate on test set"""
        logger.info("Evaluating on test set...")
        
        test_data = {
            station: data["test"]
            for station, data in self.processed_data.items()
        }
        
        test_metrics = self.trainer.evaluate_global_model(test_data)
        
        logger.info("=== Test Results ===")
        for client_id, metrics in test_metrics.items():
            if client_id != "overall":
                logger.info(
                    f"Client {client_id}: MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}"
                )
        
        logger.info(
            f"Overall: MAE={test_metrics['overall']['mae']:.4f}, "
            f"RMSE={test_metrics['overall']['rmse']:.4f}"
        )
        
        return test_metrics
    
    def save_results(self):
        """Save training results and plots"""
        logger.info("Saving results...")
        
        # Save training history
        history_path = OUTPUT_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        if self.training_history:
            losses = [r["avg_client_loss"] for r in self.training_history]
            rounds = [r["round"] for r in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, 'b-', label='Average Client Loss')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.title('Federated Training Loss')
            plt.legend()
            plt.grid(True)
            
            plot_path = OUTPUT_DIR / "training_curve.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved training curve to {plot_path}")
        
        logger.info(f"Results saved to {OUTPUT_DIR}")


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("FedAIR Federated Learning Training Pipeline")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = FederatedTrainingPipeline()
    
    # Prepare data
    pipeline.prepare_data()
    
    # Train
    pipeline.train()
    
    # Evaluate
    test_metrics = pipeline.evaluate()
    
    # Save results
    pipeline.save_results()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

