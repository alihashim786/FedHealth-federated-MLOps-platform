"""
Feature Engineering Module
Creates temporal features, lag features, and sliding windows for sequence modeling
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN,
    LAG_FEATURES, SEQUENCE_LENGTH, PREDICTION_HORIZON
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates temporal and lag features for time series forecasting
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from timestamp
        
        Args:
            df: Input DataFrame with timestamp column
        
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        if TIMESTAMP_COLUMN not in df.columns:
            raise ValueError(f"Timestamp column '{TIMESTAMP_COLUMN}' not found")
        
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        
        # Extract temporal features
        df['hour'] = df[TIMESTAMP_COLUMN].dt.hour
        df['day_of_week'] = df[TIMESTAMP_COLUMN].dt.dayofweek
        df['day_of_month'] = df[TIMESTAMP_COLUMN].dt.day
        df['month'] = df[TIMESTAMP_COLUMN].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour and day_of_week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        temporal_features = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
        ]
        
        logger.info(f"Created {len(temporal_features)} temporal features")
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lag features for all feature columns
        
        Args:
            df: Input DataFrame (should be sorted by timestamp)
            lags: List of lag hours (default: from config)
        
        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = LAG_FEATURES
        
        df = df.copy()
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        
        lag_feature_names = []
        
        for feature in FEATURE_COLUMNS:
            for lag in lags:
                lag_col = f"{feature}_lag_{lag}h"
                df[lag_col] = df[feature].shift(lag)
                lag_feature_names.append(lag_col)
        
        # Drop rows with NaN from lag features
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Created {len(lag_feature_names)} lag features")
        self.feature_names.extend(lag_feature_names)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            windows: List of window sizes in hours
        
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        
        rolling_features = []
        
        for feature in FEATURE_COLUMNS:
            for window in windows:
                # Rolling mean
                df[f"{feature}_rolling_mean_{window}h"] = df[feature].rolling(window=window, min_periods=1).mean()
                # Rolling std
                df[f"{feature}_rolling_std_{window}h"] = df[feature].rolling(window=window, min_periods=1).std()
                # Rolling max
                df[f"{feature}_rolling_max_{window}h"] = df[feature].rolling(window=window, min_periods=1).max()
                # Rolling min
                df[f"{feature}_rolling_min_{window}h"] = df[feature].rolling(window=window, min_periods=1).min()
                
                rolling_features.extend([
                    f"{feature}_rolling_mean_{window}h",
                    f"{feature}_rolling_std_{window}h",
                    f"{feature}_rolling_max_{window}h",
                    f"{feature}_rolling_min_{window}h"
                ])
        
        # Fill NaN from rolling windows
        df[rolling_features] = df[rolling_features].bfill().ffill()
        
        logger.info(f"Created {len(rolling_features)} rolling features")
        self.feature_names.extend(rolling_features)
        
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True,
        include_lags: bool = True,
        include_rolling: bool = False
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            include_temporal: Whether to add temporal features
            include_lags: Whether to add lag features
            include_rolling: Whether to add rolling features
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        if include_temporal:
            df = self.create_temporal_features(df)
        
        if include_lags:
            df = self.create_lag_features(df)
        
        if include_rolling:
            df = self.create_rolling_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df


class SequenceBuilder:
    """
    Builds sliding window sequences for sequence-to-sequence modeling
    """
    
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        prediction_horizon: int = PREDICTION_HORIZON
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences
        
        Args:
            df: Input DataFrame (should be sorted by timestamp)
            feature_cols: List of feature column names
            target_col: Target column name
        
        Returns:
            Tuple of (X, y) where:
            - X: (n_samples, sequence_length, n_features) array
            - y: (n_samples, prediction_horizon) array
        """
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        
        # Prepare feature matrix
        feature_matrix = df[feature_cols].values
        target_values = df[target_col].values
        
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence: [i, i+sequence_length)
            X_seq = feature_matrix[i:i+self.sequence_length]
            
            # Target sequence: [i+sequence_length, i+sequence_length+prediction_horizon)
            y_seq = target_values[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            
            X.append(X_seq)
            y.append(y_seq)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(
            f"Created {len(X)} sequences: X shape {X.shape}, y shape {y.shape}"
        )
        
        return X, y
    
    def create_sequences_from_dataframes(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = TARGET_COLUMN
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Create sequences for train/val/test splits
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        X_train, y_train = self.create_sequences(train_df, feature_cols, target_col)
        X_val, y_val = self.create_sequences(val_df, feature_cols, target_col)
        X_test, y_test = self.create_sequences(test_df, feature_cols, target_col)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Test feature engineering
    from data_ingestion.node_simulator import NodeSimulator
    from data_preprocessing.preprocess import DataPreprocessor
    
    print("=== Testing Feature Engineering ===")
    
    simulator = NodeSimulator()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    sequence_builder = SequenceBuilder()
    
    # Test on first station
    station = "CCM"
    node_data = simulator.get_node_data(station)
    
    if not node_data.empty:
        # Preprocess
        processed = preprocessor.preprocess_node(node_data, station)
        train_df = processed["train"]
        
        # Engineer features
        train_df = feature_engineer.engineer_features(
            train_df,
            include_temporal=True,
            include_lags=True,
            include_rolling=False
        )
        
        print(f"\nFeature engineering for {station}:")
        print(f"  Original features: {len(FEATURE_COLUMNS)}")
        print(f"  Total features after engineering: {len(train_df.columns)}")
        
        # Create sequences
        feature_cols = [col for col in train_df.columns if col not in [TIMESTAMP_COLUMN, TARGET_COLUMN]]
        X, y = sequence_builder.create_sequences(train_df, feature_cols)
        print(f"  Sequences created: X shape {X.shape}, y shape {y.shape}")

