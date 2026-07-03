"""
Node Simulator for Federated Learning
Simulates 9 federated nodes (one per station) for the Nanjing dataset
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Generator, Dict, List
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DATA_ROOT, STATIONS, TIMESTAMP_COLUMN, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeSimulator:
    """
    Simulates federated learning nodes (one per station)
    Each node loads its corresponding station data
    """
    
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root
        self.stations = STATIONS
        self.node_data = {}
        self._load_all_stations()
    
    def _load_all_stations(self):
        """Load data for all stations"""
        logger.info(f"Loading data for {len(self.stations)} stations...")
        
        for station in self.stations:
            csv_path = self.data_root / f"{station}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
                df = df.sort_values(TIMESTAMP_COLUMN)
                self.node_data[station] = df
                logger.info(f"Loaded {len(df)} records for station {station}")
            else:
                logger.warning(f"File not found: {csv_path}")
                self.node_data[station] = pd.DataFrame()
    
    def get_node_data(self, station: str) -> pd.DataFrame:
        """Get data for a specific station/node"""
        return self.node_data.get(station, pd.DataFrame())
    
    def get_all_nodes(self) -> Dict[str, pd.DataFrame]:
        """Get data for all nodes"""
        return self.node_data
    
    def get_node_info(self) -> Dict[str, Dict]:
        """Get metadata for all nodes"""
        info = {}
        for station, df in self.node_data.items():
            if not df.empty:
                info[station] = {
                    "num_records": len(df),
                    "start_date": df[TIMESTAMP_COLUMN].min().isoformat(),
                    "end_date": df[TIMESTAMP_COLUMN].max().isoformat(),
                    "features": FEATURE_COLUMNS,
                    "missing_values": df[FEATURE_COLUMNS].isnull().sum().to_dict()
                }
        return info


def create_node_batches(
    node_data: pd.DataFrame,
    batch_size: int = 100,
    station_name: str = "unknown"
) -> Generator[Dict, None, None]:
    """
    Create JSON batches from node data for federated learning
    
    Args:
        node_data: DataFrame for a specific node
        batch_size: Number of samples per batch
        station_name: Name of the station/node
    
    Yields:
        Dictionary containing batch data in JSON-serializable format
    """
    if node_data.empty:
        logger.warning(f"No data available for station {station_name}")
        return
    
    num_batches = len(node_data) // batch_size + (1 if len(node_data) % batch_size > 0 else 0)
    logger.info(f"Creating {num_batches} batches for station {station_name}")
    
    for i in range(0, len(node_data), batch_size):
        batch = node_data.iloc[i:i+batch_size].copy()
        
        batch_dict = {
            "station": station_name,
            "batch_id": i // batch_size,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(batch),
            "data": batch.to_dict(orient="records")
        }
        
        yield batch_dict


if __name__ == "__main__":
    # Test the node simulator
    simulator = NodeSimulator()
    info = simulator.get_node_info()
    
    print("\n=== Node Information ===")
    for station, details in info.items():
        print(f"\nStation: {station}")
        print(f"  Records: {details['num_records']}")
        print(f"  Date Range: {details['start_date']} to {details['end_date']}")
    
    # Test batch creation
    print("\n=== Testing Batch Creation ===")
    for station in STATIONS[:2]:  # Test first 2 stations
        node_data = simulator.get_node_data(station)
        batches = list(create_node_batches(node_data, batch_size=100, station_name=station))
        print(f"Station {station}: Created {len(batches)} batches")

