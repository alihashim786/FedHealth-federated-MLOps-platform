"""
Federated Learning Aggregator
Handles secure aggregation and model synchronization
"""

import torch
import logging
from typing import List, Dict, Optional
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.fedair_server import FedAIRServer
from utils.config import SERVER_CONFIG, FEDERATED_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureAggregator:
    """
    Secure aggregation for federated learning
    Simulates secure aggregation with differential privacy
    """
    
    def __init__(self, enable_dp: bool = False, epsilon: float = 1.0, delta: float = 1e-5):
        self.enable_dp = enable_dp
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, model_state: Dict[str, torch.Tensor], sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to model parameters
        
        Args:
            model_state: Model state dictionary
            sensitivity: Sensitivity parameter for DP
        
        Returns:
            Noisy model state
        """
        if not self.enable_dp:
            return model_state
        
        # Calculate noise scale
        noise_scale = (2 * sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        
        noisy_state = {}
        for key, value in model_state.items():
            noise = torch.normal(0, noise_scale, size=value.shape, device=value.device)
            noisy_state[key] = value + noise
        
        return noisy_state
    
    def aggregate_securely(
        self,
        client_models: List[torch.nn.Module],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate client models
        
        Args:
            client_models: List of client models
            client_weights: Weights for each client
        
        Returns:
            Aggregated model state
        """
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Get first model's state structure
        aggregated_state = {}
        first_state = client_models[0].state_dict()
        
        for param_name in first_state.keys():
            aggregated_param = torch.zeros_like(first_state[param_name])
            
            for model, weight in zip(client_models, client_weights):
                param = model.state_dict()[param_name]
                
                # Add DP noise if enabled
                if self.enable_dp:
                    param_dict = {param_name: param}
                    noisy_param_dict = self.add_noise(param_dict)
                    param = noisy_param_dict[param_name]
                
                aggregated_param += weight * param
            
            aggregated_state[param_name] = aggregated_param
        
        return aggregated_state


class FederatedAggregator:
    """
    Main aggregator for federated learning
    Coordinates model aggregation using FedAIR server
    """
    
    def __init__(self, server: FedAIRServer):
        self.server = server
        self.secure_aggregator = SecureAggregator(
            enable_dp=FEDERATED_CONFIG.get("differential_privacy", False),
            epsilon=FEDERATED_CONFIG.get("dp_epsilon", 1.0),
            delta=FEDERATED_CONFIG.get("dp_delta", 1e-5)
        )
        self.aggregation_history = []
    
    def aggregate_round(
        self,
        client_models: List[torch.nn.Module],
        client_data_sizes: List[int] = None,
        client_weights: List[float] = None,
        round_num: int = 0
    ) -> torch.nn.Module:
        """
        Aggregate models from a federated learning round
        
        Args:
            client_models: List of client models to aggregate
            client_data_sizes: Optional data sizes for weighted aggregation
            client_weights: Optional explicit weights
            round_num: Current round number
        
        Returns:
            Aggregated global model
        """
        logger.info(f"Aggregating round {round_num} with {len(client_models)} clients")
        
        # Use secure aggregation if enabled
        if FEDERATED_CONFIG.get("secure_aggregation", False):
            # Apply secure aggregation
            aggregated_state = self.secure_aggregator.aggregate_securely(
                client_models,
                client_weights or [1.0 / len(client_models)] * len(client_models)
            )
            
            # Update global model
            self.server.global_model.load_state_dict(aggregated_state)
        else:
            # Use standard aggregation
            self.server.aggregate(
                client_models,
                client_weights=client_weights,
                client_data_sizes=client_data_sizes
            )
        
        # Log aggregation
        self.aggregation_history.append({
            "round": round_num,
            "num_clients": len(client_models),
            "method": self.server.aggregation_method
        })
        
        return self.server.get_global_model()
    
    def get_aggregation_stats(self) -> Dict:
        """Get statistics about aggregation history"""
        if not self.aggregation_history:
            return {}
        
        return {
            "total_rounds": len(self.aggregation_history),
            "avg_clients_per_round": np.mean([h["num_clients"] for h in self.aggregation_history]),
            "aggregation_method": self.server.aggregation_method
        }


if __name__ == "__main__":
    # Test aggregator
    print("=== Testing Federated Aggregator ===")
    
    from models.fedair_client import create_client_model
    from utils.config import CLIENT_CONFIG
    
    # Create server and aggregator
    server = FedAIRServer(**SERVER_CONFIG)
    aggregator = FederatedAggregator(server)
    
    # Create dummy client models
    client_models = [create_client_model(CLIENT_CONFIG) for _ in range(3)]
    server.initialize_global_model(client_models[0])
    
    # Test aggregation
    global_model = aggregator.aggregate_round(
        client_models,
        client_data_sizes=[100, 200, 150],
        round_num=1
    )
    
    print(f"Aggregation successful: {global_model is not None}")
    stats = aggregator.get_aggregation_stats()
    print(f"Aggregation stats: {stats}")

