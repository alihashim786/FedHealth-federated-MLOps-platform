"""
Monitoring API for Real-time PM2.5 Predictions and System Monitoring
Integrates with Prometheus/Grafana
"""

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.fedair_client import create_client_model, FedAIRClient
from utils.config import (
    MODEL_DIR, MONITORING_CONFIG, CLIENT_CONFIG, SEQUENCE_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FedAIR Monitoring API")

# Global model cache
model_cache: Dict[str, FedAIRClient] = {}
prediction_history: List[Dict] = []
latency_history: List[Dict] = []


class PredictionRequest(BaseModel):
    station: str
    features: List[List[float]]  # Sequence of feature vectors
    timestamp: Optional[str] = None


class PredictionResponse(BaseModel):
    station: str
    predictions: List[float]
    timestamp: str
    latency_ms: float
    confidence: Optional[float] = None


def load_model(station: str = "global") -> Optional[FedAIRClient]:
    """Load model for a station"""
    if station in model_cache:
        return model_cache[station]
    
    # Try to load station-specific model
    model_path = MODEL_DIR / f"global_model_round_*.pth"
    model_files = list(MODEL_DIR.glob("global_model*.pth"))
    
    if not model_files:
        logger.warning("No model files found")
        return None
    
    # Load most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    try:
        model = create_client_model(CLIENT_CONFIG)
        model.load_state_dict(torch.load(latest_model, map_location="cpu"))
        model.eval()
        model_cache[station] = model
        logger.info(f"Loaded model from {latest_model}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Initializing monitoring API...")
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FedAIR Monitoring API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = len(model_cache) > 0
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict PM2.5 for a given station and feature sequence
    
    Args:
        request: Prediction request with station and features
    
    Returns:
        Prediction response with PM2.5 forecasts
    """
    start_time = time.time()
    
    # Load model
    model = load_model(request.station)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate input
    if len(request.features) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {SEQUENCE_LENGTH} feature vectors, got {len(request.features)}"
        )
    
    # Convert to tensor
    try:
        features_tensor = torch.FloatTensor(request.features).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature format: {e}")
    
    # Make prediction
    try:
        with torch.no_grad():
            predictions = model(features_tensor)
            predictions = predictions.squeeze(0).cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log prediction
    prediction_record = {
        "station": request.station,
        "timestamp": request.timestamp or datetime.now().isoformat(),
        "latency_ms": latency_ms,
        "predictions": predictions
    }
    prediction_history.append(prediction_record)
    
    # Log latency
    latency_history.append({
        "timestamp": datetime.now().isoformat(),
        "latency_ms": latency_ms,
        "station": request.station
    })
    
    # Check latency threshold
    if latency_ms > MONITORING_CONFIG["latency_threshold_ms"]:
        logger.warning(f"High latency detected: {latency_ms:.2f}ms")
    
    response = PredictionResponse(
        station=request.station,
        predictions=predictions,
        timestamp=datetime.now().isoformat(),
        latency_ms=latency_ms
    )
    
    return response


@app.get("/metrics")
async def get_metrics():
    """Get monitoring metrics"""
    if not latency_history:
        return {
            "avg_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "total_predictions": 0
        }
    
    latencies = [h["latency_ms"] for h in latency_history[-100:]]  # Last 100 predictions
    
    return {
        "avg_latency_ms": np.mean(latencies),
        "max_latency_ms": np.max(latencies),
        "min_latency_ms": np.min(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "total_predictions": len(prediction_history),
        "recent_predictions": len(latencies)
    }


@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history"""
    return {
        "predictions": prediction_history[-limit:],
        "total": len(prediction_history)
    }


@app.get("/alerts")
async def get_alerts():
    """Get current alerts"""
    alerts = []
    
    # Check latency
    if latency_history:
        recent_latencies = [h["latency_ms"] for h in latency_history[-10:]]
        avg_latency = np.mean(recent_latencies)
        
        if avg_latency > MONITORING_CONFIG["latency_threshold_ms"]:
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"Average latency {avg_latency:.2f}ms exceeds threshold",
                "timestamp": datetime.now().isoformat()
            })
    
    # Check model availability
    if len(model_cache) == 0:
        alerts.append({
            "type": "model_unavailable",
            "severity": "critical",
            "message": "No models loaded",
            "timestamp": datetime.now().isoformat()
        })
    
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/prometheus/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    if latency_history:
        latencies = [h["latency_ms"] for h in latency_history[-100:]]
        metrics.append(f"# HELP prediction_latency_ms Prediction latency in milliseconds")
        metrics.append(f"# TYPE prediction_latency_ms gauge")
        metrics.append(f"prediction_latency_ms {np.mean(latencies)}")
        
        metrics.append(f"# HELP prediction_count_total Total number of predictions")
        metrics.append(f"# TYPE prediction_count_total counter")
        metrics.append(f"prediction_count_total {len(prediction_history)}")
    
    return "\n".join(metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MONITORING_CONFIG["api_port"]
    )

