"""
Lite Dashboard - Fast Loading Version
======================================

Minimal Streamlit dashboard that loads quickly by deferring heavy imports.
"""

import streamlit as st
import os
from pathlib import Path

# Page configuration - set BEFORE any other st calls
st.set_page_config(
    page_title="GitHub Activity Predictor - Lite",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GitHub Activity Predictor")
st.caption("Fast-loading lite version")

# Quick repo list without filesystem scan
QUICK_REPOS = [
    "3b1b__manim", "facebook__react", "microsoft__vscode",
    "tensorflow__tensorflow", "vuejs__vue", "django__django",
    "pallets__flask", "torvalds__linux", "golang__go",
    "rust-lang__rust", "kubernetes__kubernetes", "python__cpython"
]

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Try fast dir listing
    data_dir = "repositories"
    try:
        if os.path.exists(data_dir):
            repos = sorted([d for d in os.listdir(data_dir) if not d.startswith(".")])[:50]
            st.success(f"Found {len(repos)} repositories (showing first 50)")
        else:
            repos = QUICK_REPOS
            st.warning("Using sample repository list")
    except:
        repos = QUICK_REPOS
        st.warning("Using fallback repository list")
    
    selected_repo = st.selectbox("Select Repository", repos)
    horizon = st.slider("Forecast Horizon (weeks)", 1, 12, 4)
    
    run_forecast = st.button("Run Forecast", type="primary")

# Main content
if run_forecast:
    with st.spinner("Loading dependencies..."):
        import pandas as pd
        import numpy as np
        
    repo_path = f"{data_dir}/{selected_repo}"
    
    with st.spinner(f"Loading {selected_repo}..."):
        try:
            # Inline minimal data loading
            commits_path = Path(repo_path) / "commits.csv"
            if commits_path.exists():
                df = pd.read_csv(commits_path)
                df['date'] = pd.to_datetime(df.get('author_date', df.get('date', df.columns[0])))
                weekly = df.set_index('date').resample('W').size().reset_index(name='commits')
                weekly.columns = ['ds', 'commits']
                
                st.subheader(f"📈 {selected_repo.replace('__', '/')}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Weeks", len(weekly))
                col2.metric("Avg Commits/Week", f"{weekly['commits'].mean():.1f}")
                col3.metric("Max Commits/Week", int(weekly['commits'].max()))
                
                st.line_chart(weekly.set_index('ds')['commits'])
                
                st.success(f"Data loaded: {len(weekly)} weeks of commit history")
                
                # Simple forecast display
                st.info(f"Forecast horizon: {horizon} weeks from {weekly['ds'].max().strftime('%Y-%m-%d')}")
            else:
                st.error(f"No commits.csv found in {repo_path}")
        except Exception as e:
            st.error(f"Error loading data: {e}")
else:
    st.info("👆 Select a repository and click 'Run Forecast' to analyze")
    
    # Quick stats
    st.subheader("Available Features")
    cols = st.columns(4)
    cols[0].info("📊 Time Series Visualization")
    cols[1].info("🔮 Prophet Forecasting")
    cols[2].info("🧠 Neural Network Predictions")
    cols[3].info("📉 Model Comparison")
