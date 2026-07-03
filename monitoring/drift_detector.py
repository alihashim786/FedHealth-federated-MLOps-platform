"""
Drift Detection Module
Implements KL divergence and ADWIN drift detection
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Optional, Tuple
from collections import deque
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DRIFT_CONFIG, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KLDivergenceDetector:
    """
    KL Divergence-based drift detection
    Detects distribution shift using Kullback-Leibler divergence
    """
    
    def __init__(self, threshold: float = 0.1, bins: int = 50):
        self.threshold = threshold
        self.bins = bins
        self.reference_distributions = {}
    
    def fit(self, reference_data: pd.DataFrame, station: str):
        """
        Fit detector on reference data
        
        Args:
            reference_data: Reference DataFrame
            station: Station identifier
        """
        distributions = {}
        
        for col in FEATURE_COLUMNS:
            if col in reference_data.columns:
                values = reference_data[col].dropna().values
                if len(values) > 0:
                    # Create histogram
                    hist, bin_edges = np.histogram(values, bins=self.bins)
                    # Normalize to get probability distribution
                    hist = hist / hist.sum() if hist.sum() > 0 else hist
                    distributions[col] = {
                        "hist": hist,
                        "bin_edges": bin_edges
                    }
        
        self.reference_distributions[station] = distributions
        logger.info(f"Fitted KL detector for station {station}")
    
    def detect(
        self,
        current_data: pd.DataFrame,
        station: str
    ) -> Tuple[bool, float, Dict]:
        """
        Detect drift in current data
        
        Args:
            current_data: Current DataFrame to check
            station: Station identifier
        
        Returns:
            Tuple of (drift_detected, max_kl_divergence, feature_kls)
        """
        if station not in self.reference_distributions:
            logger.warning(f"No reference distribution for station {station}")
            return False, 0.0, {}
        
        reference_dists = self.reference_distributions[station]
        feature_kls = {}
        max_kl = 0.0
        
        for col in FEATURE_COLUMNS:
            if col not in reference_dists or col not in current_data.columns:
                continue
            
            current_values = current_data[col].dropna().values
            
            if len(current_values) == 0:
                continue
            
            # Create histogram for current data
            ref_bin_edges = reference_dists[col]["bin_edges"]
            current_hist, _ = np.histogram(current_values, bins=ref_bin_edges)
            current_hist = current_hist / current_hist.sum() if current_hist.sum() > 0 else current_hist
            
            # Calculate KL divergence
            ref_hist = reference_dists[col]["hist"]
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            current_hist = current_hist + epsilon
            ref_hist = ref_hist / ref_hist.sum()
            current_hist = current_hist / current_hist.sum()
            
            kl_div = stats.entropy(current_hist, ref_hist)
            feature_kls[col] = kl_div
            max_kl = max(max_kl, kl_div)
        
        drift_detected = max_kl > self.threshold
        
        if drift_detected:
            logger.warning(
                f"Drift detected for station {station}: "
                f"max_KL={max_kl:.4f} (threshold={self.threshold})"
            )
        
        return drift_detected, max_kl, feature_kls


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) drift detector
    Detects concept drift by monitoring mean and variance
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.windows = {}  # One window per station/feature
    
    def _init_window(self, station: str, feature: str):
        """Initialize window for a station-feature pair"""
        key = f"{station}_{feature}"
        if key not in self.windows:
            self.windows[key] = {
                "values": deque(),
                "mean": 0.0,
                "variance": 0.0,
                "n": 0
            }
    
    def _update_statistics(self, window: Dict, value: float):
        """Update running statistics"""
        window["n"] += 1
        n = window["n"]
        old_mean = window["mean"]
        new_mean = old_mean + (value - old_mean) / n
        window["variance"] = window["variance"] + (value - old_mean) * (value - new_mean)
        window["mean"] = new_mean
        window["values"].append(value)
    
    def _detect_cut(self, window: Dict) -> Optional[int]:
        """
        Detect cut point in window
        
        Returns:
            Cut point index if drift detected, None otherwise
        """
        values = list(window["values"])
        n = len(values)
        
        if n < 2:
            return None
        
        for i in range(1, n):
            n0 = i
            n1 = n - i
            
            mean0 = np.mean(values[:i])
            mean1 = np.mean(values[i:])
            
            # Calculate threshold
            m = 1 / (1 / n0 + 1 / n1)
            delta_prime = self.delta / n
            variance = window["variance"] / window["n"] if window["n"] > 0 else 0.0
            
            threshold = np.sqrt((1 / (2 * m)) * np.log(2 / delta_prime) * variance)
            
            if abs(mean0 - mean1) > threshold:
                return i
        
        return None
    
    def detect(
        self,
        value: float,
        station: str,
        feature: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect drift for a single value
        
        Args:
            value: Current value
            station: Station identifier
            feature: Feature name
        
        Returns:
            Tuple of (drift_detected, new_mean)
        """
        self._init_window(station, feature)
        key = f"{station}_{feature}"
        window = self.windows[key]
        
        # Update statistics
        self._update_statistics(window, value)
        
        # Check for drift
        cut_point = self._detect_cut(window)
        
        if cut_point is not None:
            # Drift detected - reset window
            logger.warning(
                f"ADWIN drift detected for {station}_{feature} at cut point {cut_point}"
            )
            window["values"] = deque(list(window["values"])[cut_point:])
            window["n"] = len(window["values"])
            if window["n"] > 0:
                window["mean"] = np.mean(window["values"])
                window["variance"] = np.var(window["values"])
            else:
                window["mean"] = 0.0
                window["variance"] = 0.0
            
            return True, window["mean"]
        
        return False, window["mean"]


class DriftDetector:
    """
    Main drift detector combining multiple methods
    """
    
    def __init__(self, method: str = "kl_divergence"):
        self.method = method
        
        if method == "kl_divergence":
            self.detector = KLDivergenceDetector(
                threshold=DRIFT_CONFIG.get("kl_threshold", 0.1)
            )
        elif method == "adwin":
            self.detector = ADWINDetector(
                delta=DRIFT_CONFIG.get("adwin_delta", 0.002)
            )
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
        
        self.drift_history = []
        self.retrain_triggered = False
    
    def fit(self, reference_data: pd.DataFrame, station: str):
        """Fit detector on reference data"""
        if self.method == "kl_divergence":
            self.detector.fit(reference_data, station)
        # ADWIN doesn't need explicit fitting
    
    def check_drift(
        self,
        current_data: pd.DataFrame,
        station: str
    ) -> Dict:
        """
        Check for drift in current data
        
        Args:
            current_data: Current data to check
            station: Station identifier
        
        Returns:
            Dictionary with drift detection results
        """
        if self.method == "kl_divergence":
            drift_detected, max_kl, feature_kls = self.detector.detect(current_data, station)
            
            result = {
                "drift_detected": drift_detected,
                "max_kl_divergence": max_kl,
                "feature_kls": feature_kls,
                "station": station,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            # ADWIN - check each feature
            drift_detected = False
            feature_drifts = {}
            
            for col in FEATURE_COLUMNS:
                if col in current_data.columns:
                    values = current_data[col].dropna()
                    for val in values:
                        drift, mean = self.detector.detect(val, station, col)
                        if drift:
                            drift_detected = True
                            feature_drifts[col] = {"drift": True, "new_mean": mean}
            
            result = {
                "drift_detected": drift_detected,
                "feature_drifts": feature_drifts,
                "station": station,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        self.drift_history.append(result)
        
        # Check if retraining should be triggered
        if result["drift_detected"]:
            if self.method == "kl_divergence":
                if result["max_kl_divergence"] > DRIFT_CONFIG.get("retrain_threshold", 0.15):
                    self.retrain_triggered = True
                    logger.warning(f"Retraining triggered for station {station}")
            else:
                self.retrain_triggered = True
        
        return result
    
    def should_retrain(self) -> bool:
        """Check if retraining should be triggered"""
        return self.retrain_triggered
    
    def reset_retrain_flag(self):
        """Reset retrain flag"""
        self.retrain_triggered = False
    
    def get_drift_history(self) -> list:
        """Get drift detection history"""
        return self.drift_history


if __name__ == "__main__":
    # Test drift detection
    print("=== Testing Drift Detection ===")
    
    # Create sample data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        "PM2.5": np.random.normal(100, 20, 1000),
        "PM10": np.random.normal(150, 30, 1000),
        "NO2": np.random.normal(50, 10, 1000)
    })
    
    # Create shifted data (drift)
    current_data = pd.DataFrame({
        "PM2.5": np.random.normal(150, 25, 500),  # Shifted mean
        "PM10": np.random.normal(200, 35, 500),   # Shifted mean
        "NO2": np.random.normal(70, 15, 500)      # Shifted mean
    })
    
    # Test KL divergence detector
    detector = DriftDetector(method="kl_divergence")
    detector.fit(reference_data, station="test_station")
    result = detector.check_drift(current_data, station="test_station")
    
    print(f"Drift detected: {result['drift_detected']}")
    print(f"Max KL divergence: {result['max_kl_divergence']:.4f}")
    print(f"Should retrain: {detector.should_retrain()}")

