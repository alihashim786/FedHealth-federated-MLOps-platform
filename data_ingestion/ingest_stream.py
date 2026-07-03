"""
Streaming Data Ingestion System
Simulates streaming ingestion via Python generator for federated learning
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Generator, Dict, Optional
from datetime import datetime, timedelta
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data_ingestion.node_simulator import NodeSimulator, create_node_batches
from utils.config import STATIONS, TIMESTAMP_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingIngester:
    """
    Simulates streaming data ingestion for federated learning
    Processes data in time-ordered batches across all nodes
    """
    
    def __init__(self, node_simulator: NodeSimulator, batch_size: int = 100):
        self.node_simulator = node_simulator
        self.batch_size = batch_size
        self.ingestion_log = []
    
    def stream_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        simulate_delay: bool = False,
        delay_seconds: float = 0.1
    ) -> Generator[Dict, None, None]:
        """
        Stream data batches from all nodes in time order
        
        Args:
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            simulate_delay: Whether to simulate real-time delay
            delay_seconds: Delay between batches in seconds
        
        Yields:
            Dictionary containing batch data with timestamp
        """
        all_batches = []
        
        # Collect batches from all nodes
        for station in STATIONS:
            node_data = self.node_simulator.get_node_data(station)
            
            if node_data.empty:
                continue
            
            # Apply date filters if provided
            if start_date:
                node_data = node_data[node_data[TIMESTAMP_COLUMN] >= pd.to_datetime(start_date)]
            if end_date:
                node_data = node_data[node_data[TIMESTAMP_COLUMN] <= pd.to_datetime(end_date)]
            
            # Create batches for this node
            for batch in create_node_batches(node_data, self.batch_size, station):
                all_batches.append(batch)
        
        # Sort batches by timestamp (from data, not batch creation time)
        all_batches.sort(key=lambda x: x["data"][0][TIMESTAMP_COLUMN] if x["data"] else "")
        
        logger.info(f"Streaming {len(all_batches)} batches from {len(STATIONS)} nodes")
        
        # Yield batches with optional delay
        for batch in all_batches:
            ingestion_time = datetime.now().isoformat()
            batch["ingestion_timestamp"] = ingestion_time
            
            # Log ingestion
            self._log_ingestion(batch)
            
            yield batch
            
            if simulate_delay:
                time.sleep(delay_seconds)
    
    def _log_ingestion(self, batch: Dict):
        """Log batch ingestion with timestamp"""
        log_entry = {
            "timestamp": batch["ingestion_timestamp"],
            "station": batch["station"],
            "batch_id": batch["batch_id"],
            "num_samples": batch["num_samples"]
        }
        self.ingestion_log.append(log_entry)
        logger.info(
            f"Ingested batch {batch['batch_id']} from {batch['station']} "
            f"({batch['num_samples']} samples)"
        )
    
    def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested data"""
        if not self.ingestion_log:
            return {"total_batches": 0, "stations": {}}
        
        stats = {
            "total_batches": len(self.ingestion_log),
            "stations": {}
        }
        
        for entry in self.ingestion_log:
            station = entry["station"]
            if station not in stats["stations"]:
                stats["stations"][station] = {
                    "batches": 0,
                    "samples": 0
                }
            stats["stations"][station]["batches"] += 1
            stats["stations"][station]["samples"] += entry["num_samples"]
        
        return stats
    
    def save_ingestion_log(self, output_path: Path):
        """Save ingestion log to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.ingestion_log, f, indent=2)
        logger.info(f"Saved ingestion log to {output_path}")


class BatchProcessor:
    """
    Processes ingested batches for federated learning
    Converts batches to format suitable for training
    """
    
    @staticmethod
    def batch_to_dataframe(batch: Dict) -> pd.DataFrame:
        """Convert batch dictionary to DataFrame"""
        return pd.DataFrame(batch["data"])
    
    @staticmethod
    def validate_batch(batch: Dict) -> bool:
        """Validate batch structure and data"""
        required_keys = ["station", "batch_id", "data", "timestamp"]
        if not all(key in batch for key in required_keys):
            return False
        
        if not batch["data"]:
            return False
        
        return True


if __name__ == "__main__":
    # Test streaming ingestion
    print("=== Testing Streaming Ingestion ===")
    
    simulator = NodeSimulator()
    ingester = StreamingIngester(simulator, batch_size=100)
    
    # Stream a few batches
    batch_count = 0
    for batch in ingester.stream_data(simulate_delay=False):
        print(f"Batch {batch['batch_id']} from {batch['station']}: {batch['num_samples']} samples")
        batch_count += 1
        if batch_count >= 5:  # Test with 5 batches
            break
    
    # Get statistics
    stats = ingester.get_ingestion_stats()
    print(f"\n=== Ingestion Statistics ===")
    print(f"Total batches: {stats['total_batches']}")
    for station, details in stats["stations"].items():
        print(f"{station}: {details['batches']} batches, {details['samples']} samples")

