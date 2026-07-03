"""
Health Authority Dashboard
Displays pollution heatmap, risk scores, forecasts, and anomaly alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import STATIONS, MONITORING_CONFIG, DASHBOARD_CONFIG
from data_ingestion.node_simulator import NodeSimulator

# Page config
st.set_page_config(
    page_title="FedAIR Health Authority Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "node_simulator" not in st.session_state:
    st.session_state.node_simulator = NodeSimulator()

# Title
st.title("🏥 Health Authority Dashboard")
st.markdown("Real-time Air Quality Monitoring and Forecasting System")

# Sidebar
st.sidebar.header("Dashboard Controls")
selected_station = st.sidebar.selectbox("Select Station", STATIONS)
forecast_hours = st.sidebar.selectbox("Forecast Horizon", [24, 48], index=0)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 60, 600, 300)

if auto_refresh:
    st.rerun()

# Main content
col1, col2, col3, col4 = st.columns(4)

# Get latest data
node_data = st.session_state.node_simulator.get_node_data(selected_station)
if not node_data.empty:
    latest = node_data.iloc[-1]
    
    with col1:
        st.metric("PM2.5", f"{latest['PM2.5']:.1f} µg/m³", 
                 delta=f"{latest['PM2.5'] - node_data.iloc[-2]['PM2.5']:.1f}" if len(node_data) > 1 else None)
    
    with col2:
        st.metric("PM10", f"{latest['PM10']:.1f} µg/m³",
                 delta=f"{latest['PM10'] - node_data.iloc[-2]['PM10']:.1f}" if len(node_data) > 1 else None)
    
    with col3:
        st.metric("AQI", f"{latest['AQI']:.0f}",
                 delta=f"{latest['AQI'] - node_data.iloc[-2]['AQI']:.0f}" if len(node_data) > 1 else None)
    
    with col4:
        # Risk score calculation
        risk_score = 0
        if latest['PM2.5'] > 150:
            risk_score = "High"
        elif latest['PM2.5'] > 75:
            risk_score = "Moderate"
        else:
            risk_score = "Low"
        
        st.metric("Risk Level", risk_score)

# Pollution Heatmap
st.header("📊 Pollution Heatmap Across Stations")

# Create heatmap data
heatmap_data = []
for station in STATIONS:
    station_data = st.session_state.node_simulator.get_node_data(station)
    if not station_data.empty:
        latest = station_data.iloc[-1]
        heatmap_data.append({
            "Station": station,
            "PM2.5": latest['PM2.5'],
            "PM10": latest['PM10'],
            "NO2": latest['NO2'],
            "O3": latest['O3'],
            "CO": latest['CO'],
            "SO2": latest['SO2']
        })

if heatmap_data:
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_df.set_index("Station").T,
        labels=dict(x="Station", y="Pollutant", color="Concentration"),
        title="Current Pollution Levels by Station",
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

# Forecast Section
st.header(f"🔮 {forecast_hours}-Hour Forecast")

# Simulate forecast (in production, this would call the API)
if not node_data.empty:
    # Get recent data for forecast
    recent_data = node_data.tail(24)  # Last 24 hours
    
    # Create forecast visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=recent_data['pubtime'],
        y=recent_data['PM2.5'],
        mode='lines',
        name='Historical PM2.5',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast (simulated)
    forecast_start = recent_data['pubtime'].iloc[-1]
    forecast_times = [forecast_start + timedelta(hours=i) for i in range(1, forecast_hours + 1)]
    
    # Simple trend-based forecast (replace with actual model prediction)
    last_value = recent_data['PM2.5'].iloc[-1]
    trend = (recent_data['PM2.5'].iloc[-1] - recent_data['PM2.5'].iloc[-6]) / 6 if len(recent_data) >= 6 else 0
    forecast_values = [last_value + trend * i + np.random.normal(0, 5) for i in range(1, forecast_hours + 1)]
    
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast_values,
        mode='lines',
        name='Forecast PM2.5',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add threshold lines
    fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="High Risk Threshold")
    fig.add_hline(y=75, line_dash="dot", line_color="yellow", annotation_text="Moderate Risk Threshold")
    
    fig.update_layout(
        title=f"PM2.5 Forecast for {selected_station}",
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Anomaly Alerts
st.header("🚨 Anomaly Alerts")

# Check for anomalies
alerts = []
for station in STATIONS:
    station_data = st.session_state.node_simulator.get_node_data(station)
    if not station_data.empty:
        latest = station_data.iloc[-1]
        
        if latest['PM2.5'] > MONITORING_CONFIG.get("alert_threshold_pm25", 150):
            alerts.append({
                "station": station,
                "type": "High PM2.5",
                "value": latest['PM2.5'],
                "severity": "High" if latest['PM2.5'] > 200 else "Moderate",
                "timestamp": latest['pubtime']
            })

if alerts:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.success("✅ No active alerts - All stations within safe limits")

# Risk Scores Table
st.header("📋 Risk Scores by Station")

risk_data = []
for station in STATIONS:
    station_data = st.session_state.node_simulator.get_node_data(station)
    if not station_data.empty:
        latest = station_data.iloc[-1]
        
        # Calculate risk score
        pm25 = latest['PM2.5']
        if pm25 > 150:
            risk = "High"
            risk_score = 3
        elif pm25 > 75:
            risk = "Moderate"
            risk_score = 2
        else:
            risk = "Low"
            risk_score = 1
        
        risk_data.append({
            "Station": station,
            "PM2.5": f"{pm25:.1f}",
            "PM10": f"{latest['PM10']:.1f}",
            "AQI": f"{latest['AQI']:.0f}",
            "Risk Level": risk,
            "Risk Score": risk_score,
            "Last Update": latest['pubtime']
        })

if risk_data:
    risk_df = pd.DataFrame(risk_data)
    risk_df = risk_df.sort_values("Risk Score", ascending=False)
    st.dataframe(risk_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

