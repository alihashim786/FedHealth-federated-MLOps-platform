"""
Federated Learning Trainer
Implements client update loop, communication rounds, and periodic evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import mlflow
import mlflow.pytorch

sys.path.append(str(Path(__file__).parent.parent))
from models.fedair_client import FedAIRClient, create_client_model
from models.fedair_server import FedAIRServer, create_server
from fed_learning.aggregator import FederatedAggregator
from utils.config import (
    FEDERATED_CONFIG, TRAINING_CONFIG, CLIENT_CONFIG, SERVER_CONFIG,
    MLFLOW_CONFIG, MODEL_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientTrainer:
    """
    Trains a client model locally
    """
    
    def __init__(
        self,
        model: FedAIRClient,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=FEDERATED_CONFIG["learning_rate"],
            weight_decay=FEDERATED_CONFIG.get("weight_decay", 0.0001)
        )
        self.criterion = nn.MSELoss()
        self.train_losses = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        local_epochs: int = 1
    ) -> float:
        """
        Train client model for local epochs
        
        Args:
            train_loader: DataLoader for training data
            local_epochs: Number of local training epochs
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                
                # Loss calculation (predicting future horizon)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(train_loader)
        
        avg_loss = total_loss / local_epochs
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Tuple[float, Dict]:
        """
        Evaluate client model
        
        Args:
            data_loader: DataLoader for evaluation data
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Calculate per-horizon metrics (for multi-step prediction)
        metrics = {
            "loss": avg_loss,
            "mae": mae,
            "rmse": rmse,
            "predictions": predictions,
            "targets": targets
        }
        
        return avg_loss, metrics


class FederatedTrainer:
    """
    Main federated learning trainer
    Coordinates training across multiple clients
    """
    
    def __init__(
        self,
        server: FedAIRServer,
        aggregator: FederatedAggregator,
        device: str = "cpu"
    ):
        self.server = server
        self.aggregator = aggregator
        self.device = device
        self.round_history = []
        
        # Initialize MLflow (optional - use file-based if server not available)
        try:
            mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
            mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
            self.mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow server not available, using file-based tracking: {e}")
            # Use file-based tracking instead
            mlflow.set_tracking_uri("file:./mlruns")
            try:
                mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
                self.mlflow_enabled = True
            except Exception:
                logger.warning("MLflow disabled - continuing without experiment tracking")
                self.mlflow_enabled = False
    
    def select_clients(
        self,
        all_clients: List[str],
        num_clients: int
    ) -> List[str]:
        """
        Select clients for a federated round
        
        Args:
            all_clients: List of all client IDs
            num_clients: Number of clients to select
        
        Returns:
            List of selected client IDs
        """
        selected = np.random.choice(all_clients, size=num_clients, replace=False).tolist()
        return selected
    
    def federated_round(
        self,
        client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        round_num: int,
        local_epochs: int = None
    ) -> Dict:
        """
        Execute one federated learning round
        
        Args:
            client_data: Dictionary mapping client_id to (X_train, y_train) tuples
            round_num: Current round number
            local_epochs: Number of local training epochs
        
        Returns:
            Dictionary with round results
        """
        if local_epochs is None:
            local_epochs = FEDERATED_CONFIG["local_epochs"]
        
        logger.info(f"Starting federated round {round_num}")
        
        # Select clients
        all_client_ids = list(client_data.keys())
        num_clients = min(
            FEDERATED_CONFIG["num_clients_per_round"],
            len(all_client_ids)
        )
        selected_clients = self.select_clients(all_client_ids, num_clients)
        
        logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
        
        # Train each selected client
        client_models = []
        client_data_sizes = []
        client_losses = []
        
        for client_id in selected_clients:
            X_train, y_train = client_data[client_id]
            
            # Create client model (copy of global model)
            # Update input_dim based on actual data
            client_config = CLIENT_CONFIG.copy()
            client_config["input_dim"] = X_train.shape[-1]  # Use actual feature dimension
            client_model = create_client_model(client_config)
            # Only load compatible weights
            try:
                client_model.load_state_dict(self.server.global_model.state_dict(), strict=False)
            except:
                # If dimensions don't match, just use new model
                pass
            
            # Create data loader
            dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            train_loader = DataLoader(
                dataset,
                batch_size=FEDERATED_CONFIG["batch_size"],
                shuffle=True,
                num_workers=TRAINING_CONFIG.get("num_workers", 0),
                pin_memory=TRAINING_CONFIG.get("pin_memory", False)
            )
            
            # Train client
            trainer = ClientTrainer(client_model, device=self.device)
            train_loss = trainer.train_epoch(train_loader, local_epochs=local_epochs)
            
            client_models.append(client_model)
            client_data_sizes.append(len(X_train))
            client_losses.append(train_loss)
            
            logger.info(f"Client {client_id}: train_loss={train_loss:.4f}")
        
        # Aggregate models
        global_model = self.aggregator.aggregate_round(
            client_models,
            client_data_sizes=client_data_sizes,
            round_num=round_num
        )
        
        # Log to MLflow (if enabled)
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_param("round", round_num)
                    mlflow.log_param("num_clients", len(selected_clients))
                    mlflow.log_metric("avg_client_loss", np.mean(client_losses))
                    mlflow.log_metric("std_client_loss", np.std(client_losses))
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        round_result = {
            "round": round_num,
            "selected_clients": selected_clients,
            "avg_client_loss": np.mean(client_losses),
            "client_losses": client_losses,
            "num_clients": len(selected_clients)
        }
        
        self.round_history.append(round_result)
        
        return round_result
    
    def evaluate_global_model(
        self,
        test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict:
        """
        Evaluate global model on test data from all clients
        
        Args:
            test_data: Dictionary mapping client_id to (X_test, y_test) tuples
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating global model on test data")
        
        global_model = self.server.get_global_model()
        global_model.eval()
        
        all_metrics = {}
        overall_predictions = []
        overall_targets = []
        
        for client_id, (X_test, y_test) in test_data.items():
            dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test)
            )
            test_loader = DataLoader(
                dataset,
                batch_size=FEDERATED_CONFIG["batch_size"],
                shuffle=False
            )
            
            trainer = ClientTrainer(global_model, device=self.device)
            loss, metrics = trainer.evaluate(test_loader)
            
            all_metrics[client_id] = {
                "loss": loss,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"]
            }
            
            overall_predictions.append(metrics["predictions"])
            overall_targets.append(metrics["targets"])
        
        # Overall metrics
        overall_predictions = np.concatenate(overall_predictions, axis=0)
        overall_targets = np.concatenate(overall_targets, axis=0)
        
        overall_mae = np.mean(np.abs(overall_predictions - overall_targets))
        overall_rmse = np.sqrt(np.mean((overall_predictions - overall_targets) ** 2))
        
        all_metrics["overall"] = {
            "mae": overall_mae,
            "rmse": overall_rmse
        }
        
        logger.info(f"Overall Test MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}")
        
        return all_metrics
    
    def save_model(self, path: Path, round_num: int = None):
        """Save global model"""
        model_path = path / f"global_model_round_{round_num}.pth" if round_num else path / "global_model.pth"
        torch.save(self.server.global_model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Also save with MLflow (if enabled)
        if self.mlflow_enabled and MLFLOW_CONFIG.get("log_models", True):
            try:
                mlflow.pytorch.log_model(
                    self.server.global_model,
                    "model",
                    registered_model_name="FedAIR_Global_Model"
                )
            except Exception as e:
                logger.warning(f"MLflow model logging failed: {e}")
    
    def get_training_history(self) -> Dict:
        """Get training history"""
        return {
            "rounds": self.round_history,
            "num_rounds": len(self.round_history)
        }


if __name__ == "__main__":
    # Test federated trainer
    print("=== Testing Federated Trainer ===")
    
    # Create server and aggregator
    server = create_server(SERVER_CONFIG)
    aggregator = FederatedAggregator(server)
    
    # Initialize global model
    dummy_client_model = create_client_model(CLIENT_CONFIG)
    server.initialize_global_model(dummy_client_model)
    
    # Create trainer
    trainer = FederatedTrainer(server, aggregator, device="cpu")
    
    print("Federated trainer initialized successfully")

