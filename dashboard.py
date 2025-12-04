"""
dashboard.py - Streamlit Real-Time Monitoring Dashboard

This dashboard displays:
1. Total Transactions processed
2. Fraud Count (highlighted in red)
3. Average Fraud Probability
4. Line chart of fraud probability over time

Auto-refreshes every 2 seconds to show live data.

Run with: streamlit run dashboard.py
"""

import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_FILE = "logs/api_logs.csv"
REFRESH_INTERVAL = 2000  # milliseconds (2 seconds)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="",
    layout="wide"
)

# Auto-refresh the page every 2 seconds
st_autorefresh(interval=REFRESH_INTERVAL, key="data_refresh")

# ---------------------------------------------------------------------------
# Dashboard Title
# ---------------------------------------------------------------------------
st.title("Fraud Detection - Real-Time Monitoring")
st.markdown("---")

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
def load_data():
    """
    Load prediction logs from CSV file.
    
    Returns empty DataFrame if file doesn't exist or is empty.
    """
    if not os.path.exists(LOGS_FILE):
        return pd.DataFrame(columns=['timestamp', 'prediction', 'probability'])
    
    try:
        df = pd.read_csv(LOGS_FILE)
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'prediction', 'probability'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['timestamp', 'prediction', 'probability'])


# Load the data
df = load_data()

# ---------------------------------------------------------------------------
# Key Metrics
# ---------------------------------------------------------------------------
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    total_transactions = len(df)
    st.metric(
        label="Total Transactions",
        value=f"{total_transactions:,}"
    )

with col2:
    fraud_count = df['prediction'].sum() if not df.empty else 0
    # Use delta_color to highlight fraud in red
    st.metric(
        label="Fraud Detected",
        value=f"{int(fraud_count):,}",
        delta=f"{int(fraud_count)} alerts" if fraud_count > 0 else "No fraud",
        delta_color="inverse"  # Red when positive (fraud detected)
    )

with col3:
    avg_probability = df['probability'].mean() * 100 if not df.empty else 0
    st.metric(
        label="Avg Fraud Probability",
        value=f"{avg_probability:.2f}%"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Fraud Probability Over Time Chart
# ---------------------------------------------------------------------------
st.subheader("Fraud Probability Over Time")

if not df.empty:
    # Prepare data for chart
    chart_data = df[['timestamp', 'probability']].copy()
    chart_data = chart_data.set_index('timestamp')
    chart_data['probability'] = chart_data['probability'] * 100  # Convert to percentage
    
    # Display line chart
    st.line_chart(
        chart_data,
        y='probability',
        use_container_width=True
    )
else:
    st.info("No data available yet. Start the API and send some transactions.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Recent Transactions Table
# ---------------------------------------------------------------------------
st.subheader("Recent Transactions")

if not df.empty:
    # Show last 10 transactions, newest first
    recent_df = df.tail(10).iloc[::-1].copy()
    
    # Format for display
    recent_df['Status'] = recent_df['prediction'].apply(
        lambda x: "FRAUD" if x == 1 else "Normal"
    )
    recent_df['Probability'] = recent_df['probability'].apply(
        lambda x: f"{x*100:.2f}%"
    )
    recent_df['Time'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
    
    # Display table
    display_df = recent_df[['Time', 'Status', 'Probability']]
    
    # Style the table - highlight fraud rows
    def highlight_fraud(row):
        if row['Status'] == 'FRAUD':
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_fraud, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No transactions recorded yet.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Dashboard auto-refreshes every {REFRESH_INTERVAL//1000} seconds | Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
