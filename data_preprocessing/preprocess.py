"""
Data Preprocessing Module
Handles missing values, normalization, and train/val/test splits per node
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses data for federated learning
    Handles missing values, normalization per node
    """
    
    def __init__(self, normalization_method: str = "standard"):
        """
        Args:
            normalization_method: "standard" (z-score) or "minmax" (0-1 scaling)
        """
        self.normalization_method = normalization_method
        self.scalers = {}  # One scaler per node
        self.stats = {}  # Statistics per node
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: "forward_fill", "backward_fill", "interpolate", or "mean"
        
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if method == "forward_fill":
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill()
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill()  # Fill remaining with backward
        elif method == "backward_fill":
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill()
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill()
        elif method == "interpolate":
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].interpolate(method='linear')
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill()
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill()
        elif method == "mean":
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
        
        missing_count = df[FEATURE_COLUMNS].isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Still {missing_count} missing values after handling")
        
        return df
    
    def normalize_per_node(
        self,
        df: pd.DataFrame,
        station: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize data per node (station)
        Each node has its own scaler to preserve local distribution
        
        Args:
            df: Input DataFrame
            station: Station name (node identifier)
            fit: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if fit:
            if self.normalization_method == "standard":
                scaler = StandardScaler()
            elif self.normalization_method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
            self.scalers[station] = scaler
            
            # Store statistics
            self.stats[station] = {
                "mean": scaler.mean_ if hasattr(scaler, 'mean_') else None,
                "std": scaler.scale_ if hasattr(scaler, 'scale_') else None,
                "min": scaler.data_min_ if hasattr(scaler, 'data_min_') else None,
                "max": scaler.data_max_ if hasattr(scaler, 'data_max_') else None
            }
        else:
            if station not in self.scalers:
                raise ValueError(f"Scaler not found for station {station}. Fit first.")
            scaler = self.scalers[station]
            df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
        
        return df
    
    def split_train_val_test(
        self,
        df: pd.DataFrame,
        station: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets per node
        Uses temporal split (no shuffling to preserve time order)
        
        Args:
            df: Input DataFrame (should be sorted by timestamp)
            station: Station name
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(
            f"Station {station}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def preprocess_node(
        self,
        df: pd.DataFrame,
        station: str,
        handle_missing: bool = True,
        normalize: bool = True,
        split: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline for a node
        
        Args:
            df: Raw DataFrame for the node
            station: Station name
            handle_missing: Whether to handle missing values
            normalize: Whether to normalize
            split: Whether to split into train/val/test
        
        Returns:
            Dictionary with processed DataFrames
        """
        logger.info(f"Preprocessing data for station {station}")
        
        # Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df, method="interpolate")
        
        # Normalize
        if normalize:
            df = self.normalize_per_node(df, station, fit=True)
        
        # Split
        if split:
            train_df, val_df, test_df = self.split_train_val_test(df, station)
            return {
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "all": df
            }
        else:
            return {"all": df}
    
    def save_scalers(self, output_path: Path):
        """Save scalers to disk"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {output_path}")
    
    def load_scalers(self, input_path: Path):
        """Load scalers from disk"""
        with open(input_path, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Loaded scalers from {input_path}")


class DriftDetectionPreprocessor:
    """
    Preprocessing hooks for drift detection
    Tracks data statistics for drift detection
    """
    
    def __init__(self):
        self.reference_stats = {}
    
    def compute_statistics(self, df: pd.DataFrame, station: str) -> Dict:
        """Compute statistics for drift detection"""
        stats = {
            "mean": df[FEATURE_COLUMNS].mean().to_dict(),
            "std": df[FEATURE_COLUMNS].std().to_dict(),
            "min": df[FEATURE_COLUMNS].min().to_dict(),
            "max": df[FEATURE_COLUMNS].max().to_dict(),
            "median": df[FEATURE_COLUMNS].median().to_dict()
        }
        return stats
    
    def set_reference(self, df: pd.DataFrame, station: str):
        """Set reference statistics for drift detection"""
        self.reference_stats[station] = self.compute_statistics(df, station)
        logger.info(f"Set reference statistics for station {station}")
    
    def get_reference(self, station: str) -> Optional[Dict]:
        """Get reference statistics for a station"""
        return self.reference_stats.get(station)


if __name__ == "__main__":
    # Test preprocessing
    from data_ingestion.node_simulator import NodeSimulator
    
    print("=== Testing Data Preprocessing ===")
    
    simulator = NodeSimulator()
    preprocessor = DataPreprocessor(normalization_method="standard")
    
    # Test on first station
    station = "CCM"
    node_data = simulator.get_node_data(station)
    
    if not node_data.empty:
        processed = preprocessor.preprocess_node(node_data, station)
        print(f"\nProcessed data for {station}:")
        print(f"  Train: {len(processed['train'])} samples")
        print(f"  Val: {len(processed['val'])} samples")
        print(f"  Test: {len(processed['test'])} samples")
        print(f"  Scaler saved: {station in preprocessor.scalers}")

