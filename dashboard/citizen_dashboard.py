"""
Citizen Dashboard
Provides personalized alerts, historical trends, and air quality recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import STATIONS, DASHBOARD_CONFIG
from data_ingestion.node_simulator import NodeSimulator

# Page config
st.set_page_config(
    page_title="FedAIR Citizen Dashboard",
    page_icon="👥",
    layout="wide"
)

# Initialize session state
if "node_simulator" not in st.session_state:
    st.session_state.node_simulator = NodeSimulator()

# Title
st.title("👥 Citizen Air Quality Dashboard")
st.markdown("Personalized Air Quality Information and Recommendations")

# User preferences sidebar
st.sidebar.header("Your Preferences")
user_station = st.sidebar.selectbox("Your Location", STATIONS)
alert_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
alert_threshold = st.sidebar.slider("Alert Threshold (PM2.5)", 50, 200, 100)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📍 Current Air Quality at Your Location")
    
    # Get current data
    node_data = st.session_state.node_simulator.get_node_data(user_station)
    
    if not node_data.empty:
        latest = node_data.iloc[-1]
        
        # Air quality status
        pm25 = latest['PM2.5']
        if pm25 > 150:
            status = "🔴 Unhealthy"
            status_color = "red"
            recommendation = "Avoid outdoor activities. Stay indoors with air purifier if possible."
        elif pm25 > 75:
            status = "🟡 Moderate"
            status_color = "orange"
            recommendation = "Sensitive groups should reduce outdoor activities."
        else:
            status = "🟢 Good"
            status_color = "green"
            recommendation = "Air quality is acceptable. Normal activities are fine."
        
        st.markdown(f"### {status}")
        st.info(recommendation)
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("PM2.5", f"{pm25:.1f} µg/m³")
        
        with metric_col2:
            st.metric("PM10", f"{latest['PM10']:.1f} µg/m³")
        
        with metric_col3:
            st.metric("AQI", f"{latest['AQI']:.0f}")

with col2:
    st.header("🔔 Your Alerts")
    
    if alert_enabled and not node_data.empty:
        latest = node_data.iloc[-1]
        pm25 = latest['PM2.5']
        
        if pm25 > alert_threshold:
            st.error(f"⚠️ Alert: PM2.5 is {pm25:.1f} µg/m³ (threshold: {alert_threshold})")
            st.markdown("**Action:** Consider reducing outdoor activities")
        else:
            st.success(f"✅ All good! PM2.5 is {pm25:.1f} µg/m³")
    else:
        st.info("Alerts are disabled")

# Historical Trends
st.header("📈 Historical Trends")

if not node_data.empty:
    # Get last 7 days of data
    recent_data = node_data.tail(168)  # 7 days * 24 hours
    
    # Create trend chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent_data['pubtime'],
        y=recent_data['PM2.5'],
        mode='lines',
        name='PM2.5',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,100,255,0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data['pubtime'],
        y=recent_data['PM10'],
        mode='lines',
        name='PM10',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    # Add threshold lines
    fig.add_hline(y=alert_threshold, line_dash="dot", line_color="orange", 
                  annotation_text=f"Your Alert Threshold ({alert_threshold})")
    fig.add_hline(y=150, line_dash="dot", line_color="red", 
                  annotation_text="Unhealthy Threshold (150)")
    
    fig.update_layout(
        title=f"7-Day Air Quality Trend at {user_station}",
        xaxis_title="Date",
        yaxis_title="Concentration (µg/m³)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Comparison with other stations
st.header("🗺️ Comparison with Other Stations")

comparison_data = []
for station in STATIONS:
    station_data = st.session_state.node_simulator.get_node_data(station)
    if not station_data.empty:
        latest = station_data.iloc[-1]
        comparison_data.append({
            "Station": station,
            "PM2.5": latest['PM2.5'],
            "PM10": latest['PM10'],
            "AQI": latest['AQI']
        })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    
    # Highlight user's station
    comparison_df['Is Your Station'] = comparison_df['Station'] == user_station
    
    fig = px.bar(
        comparison_df,
        x='Station',
        y='PM2.5',
        color='Is Your Station',
        title="Current PM2.5 Levels Across All Stations",
        color_discrete_map={True: 'orange', False: 'lightblue'},
        labels={'PM2.5': 'PM2.5 (µg/m³)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Recommendations
st.header("💡 Air Quality Tips")

tips = {
    "Good (0-50)": [
        "✅ Perfect day for outdoor activities",
        "✅ Open windows for fresh air",
        "✅ Great for exercise outdoors"
    ],
    "Moderate (51-100)": [
        "⚠️ Sensitive individuals should limit outdoor activities",
        "⚠️ Consider wearing a mask if you have respiratory issues",
        "✅ Generally safe for most people"
    ],
    "Unhealthy (101-150)": [
        "🔴 Everyone should reduce outdoor activities",
        "🔴 Sensitive groups should stay indoors",
        "🔴 Use air purifiers if available",
        "🔴 Keep windows closed"
    ],
    "Very Unhealthy (151+)": [
        "🔴 Avoid all outdoor activities",
        "🔴 Stay indoors with air purifier",
        "🔴 Wear N95 mask if going outside is necessary",
        "🔴 Keep all windows and doors closed"
    ]
}

current_pm25 = node_data.iloc[-1]['PM2.5'] if not node_data.empty else 0

if current_pm25 <= 50:
    category = "Good (0-50)"
elif current_pm25 <= 100:
    category = "Moderate (51-100)"
elif current_pm25 <= 150:
    category = "Unhealthy (101-150)"
else:
    category = "Very Unhealthy (151+)"

st.subheader(f"Recommendations for {category}")

for tip in tips[category]:
    st.markdown(tip)

# Daily pattern
st.header("📅 Daily Pattern")

if not node_data.empty:
    # Get data for pattern analysis
    recent_data = node_data.tail(168)  # Last week
    recent_data['hour'] = pd.to_datetime(recent_data['pubtime']).dt.hour
    
    hourly_avg = recent_data.groupby('hour')['PM2.5'].mean().reset_index()
    
    fig = px.line(
        hourly_avg,
        x='hour',
        y='PM2.5',
        title="Average PM2.5 by Hour of Day (Last 7 Days)",
        labels={'hour': 'Hour of Day', 'PM2.5': 'PM2.5 (µg/m³)'},
        markers=True
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find best and worst hours
    best_hour = hourly_avg.loc[hourly_avg['PM2.5'].idxmin(), 'hour']
    worst_hour = hourly_avg.loc[hourly_avg['PM2.5'].idxmax(), 'hour']
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🌅 Best time: {int(best_hour)}:00 ({hourly_avg.loc[hourly_avg['PM2.5'].idxmin(), 'PM2.5']:.1f} µg/m³)")
    with col2:
        st.warning(f"🌆 Worst time: {int(worst_hour)}:00 ({hourly_avg.loc[hourly_avg['PM2.5'].idxmax(), 'PM2.5']:.1f} µg/m³)")

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("Data provided by FedAIR Federated Learning System")

