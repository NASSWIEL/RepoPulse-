"""
Inference Dashboard - GitHub Activity Predictor
================================================

Enhanced Streamlit dashboard with:
- GitHub URL input for real-time prediction
- Automatic data extraction and aggregation
- Neural network inference with saved checkpoints
- Visualization of predictions

Usage:
    streamlit run src/inference_dashboard.py
"""

import os
import sys
import re
import json
import pickle
import requests
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="GitHub Commit Predictor - Neural Network",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
CHECKPOINT_DIR = Path("checkpoints")
PLOTS_DIR = Path("training_plots")


def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def parse_github_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse GitHub URL to extract owner and repo name.
    
    Args:
        url: GitHub repository URL.
        
    Returns:
        Tuple of (owner, repo_name) or (None, None) if invalid.
    """
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$",
        r"^([^/]+)/([^/]+)$"
    ]
    
    url = url.strip().rstrip("/")
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
    
    return None, None


class GitHubDataFetcher:
    """Fetches and processes data from GitHub API."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the fetcher.
        
        Args:
            token: GitHub API token (optional but recommended).
        """
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_repo_info(self, owner: str, repo: str) -> Dict:
        """Get basic repository information."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise ValueError(f"Repository not found: {owner}/{repo}")
        
        return response.json()
    
    def get_commits(
        self, 
        owner: str, 
        repo: str, 
        since: Optional[datetime] = None,
        max_pages: int = 10
    ) -> List[Dict]:
        """Fetch commit history."""
        commits = []
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        
        params = {"per_page": 100}
        if since:
            params["since"] = since.isoformat()
        
        for page in range(1, max_pages + 1):
            params["page"] = page
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                break
            
            page_commits = response.json()
            if not page_commits:
                break
            
            commits.extend(page_commits)
        
        return commits
    
    def get_pull_requests(
        self, 
        owner: str, 
        repo: str,
        state: str = "all",
        max_pages: int = 10
    ) -> List[Dict]:
        """Fetch pull request history."""
        prs = []
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        
        params = {"per_page": 100, "state": state}
        
        for page in range(1, max_pages + 1):
            params["page"] = page
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                break
            
            page_prs = response.json()
            if not page_prs:
                break
            
            prs.extend(page_prs)
        
        return prs
    
    def aggregate_weekly(self, commits: List[Dict], prs: List[Dict]) -> pd.DataFrame:
        """
        Aggregate data into weekly time series.
        
        Args:
            commits: List of commit data.
            prs: List of pull request data.
            
        Returns:
            DataFrame with weekly aggregated data.
        """
        # Process commits
        commit_dates = []
        for c in commits:
            try:
                date_str = c.get("commit", {}).get("author", {}).get("date")
                if date_str:
                    commit_dates.append(pd.to_datetime(date_str))
            except:
                pass
        
        # Process PRs
        pr_dates = []
        for pr in prs:
            try:
                date_str = pr.get("created_at")
                if date_str:
                    pr_dates.append(pd.to_datetime(date_str))
            except:
                pass
        
        if not commit_dates:
            raise ValueError("No commit data found")
        
        # Create time series
        commit_df = pd.DataFrame({"date": commit_dates})
        commit_df["date"] = pd.to_datetime(commit_df["date"]).dt.tz_localize(None)
        commit_df.set_index("date", inplace=True)
        weekly_commits = commit_df.resample("W").size()
        
        # PRs
        if pr_dates:
            pr_df = pd.DataFrame({"date": pr_dates})
            pr_df["date"] = pd.to_datetime(pr_df["date"]).dt.tz_localize(None)
            pr_df.set_index("date", inplace=True)
            weekly_prs = pr_df.resample("W").size()
        else:
            weekly_prs = pd.Series(0, index=weekly_commits.index)
        
        # Combine
        result = pd.DataFrame({
            "ds": weekly_commits.index,
            "commits": weekly_commits.values,
            "prs_opened": weekly_prs.reindex(weekly_commits.index, fill_value=0).values
        })
        
        # Sort by date and fill missing weeks
        result = result.sort_values("ds").reset_index(drop=True)
        
        return result


class NeuralNetworkInference:
    """Performs inference using saved checkpoint."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from file."""
        with open(checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)
        
        model_state = self.checkpoint["model_state"]
        
        if model_state.get("type") == "sklearn":
            self.model = model_state["model"]
            self.scaler = model_state["scaler"]
            self.use_torch = False
        else:
            # PyTorch model
            if check_torch_available():
                import torch
                import torch.nn as nn
                
                arch = model_state["model_architecture"]
                
                layers = []
                input_size = arch["input_size"]
                for hidden_size in arch["hidden_layers"]:
                    layers.append(nn.Linear(input_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(arch["dropout_rate"]))
                    input_size = hidden_size
                layers.append(nn.Linear(input_size, 1))
                
                self.model = nn.Sequential(*layers)
                self.model.load_state_dict(model_state["model_state_dict"])
                self.model.eval()
                
                self.scalers = model_state["scalers"]
                self.use_torch = True
            else:
                raise RuntimeError("PyTorch required for this checkpoint")
        
        self.lookback = model_state["lookback_window"]
        self.use_prs = model_state.get("use_prs", False)
    
    def predict(
        self, 
        commits: np.ndarray, 
        prs: Optional[np.ndarray] = None,
        horizon: int = 4
    ) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            commits: Array of historical commit counts.
            prs: Array of historical PR counts (optional).
            horizon: Number of weeks to predict.
            
        Returns:
            Dictionary with predictions and confidence intervals.
        """
        if len(commits) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} weeks of history")
        
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        # Use the last lookback values
        current_commits = list(commits[-self.lookback:])
        current_prs = list(prs[-self.lookback:]) if prs is not None and self.use_prs else None
        
        for step in range(horizon):
            # Normalize input
            if self.use_torch:
                commits_norm = self.scalers["commits"].transform(
                    np.array(current_commits).reshape(-1, 1)
                ).flatten()
                
                if current_prs is not None:
                    prs_norm = self.scalers["prs"].transform(
                        np.array(current_prs).reshape(-1, 1)
                    ).flatten()
                    X = np.concatenate([commits_norm, prs_norm]).reshape(1, -1)
                else:
                    X = commits_norm.reshape(1, -1)
                
                # Predict
                import torch
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred_norm = self.model(X_tensor).item()
                
                # Inverse transform
                pred = self.scalers["commits"].inverse_transform(
                    np.array([[pred_norm]])
                )[0, 0]
                
            else:
                # sklearn model
                commits_norm = self.scaler.transform(
                    np.array(current_commits).reshape(-1, 1)
                ).flatten()
                X = commits_norm.reshape(1, -1)
                
                pred_norm = self.model.predict(X)[0]
                pred = self.scaler.inverse_transform([[pred_norm]])[0, 0]
            
            pred = max(0, pred)
            predictions.append(pred)
            
            # Simple confidence interval estimation
            std_estimate = np.std(commits[-self.lookback:]) * (1 + 0.1 * step)
            lower_bounds.append(max(0, pred - 1.96 * std_estimate))
            upper_bounds.append(pred + 1.96 * std_estimate)
            
            # Update history for next prediction
            current_commits = current_commits[1:] + [pred]
            if current_prs is not None:
                # Assume PRs stay similar
                avg_prs = np.mean(current_prs)
                current_prs = current_prs[1:] + [avg_prs]
        
        return {
            "predictions": predictions,
            "lower": lower_bounds,
            "upper": upper_bounds
        }


def list_checkpoints() -> List[str]:
    """List available checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return []
    
    checkpoints = list(CHECKPOINT_DIR.glob("nn_model_*.pkl"))
    return [str(cp) for cp in checkpoints]


def create_prediction_plot(
    historical_data: pd.DataFrame,
    predictions: Dict[str, Any],
    repo_name: str
) -> go.Figure:
    """Create an interactive prediction plot."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data["ds"],
        y=historical_data["commits"],
        mode="lines+markers",
        name="Historical Commits",
        line=dict(color="blue", width=2),
        marker=dict(size=4)
    ))
    
    # Predictions
    last_date = historical_data["ds"].max()
    pred_dates = [last_date + timedelta(weeks=i+1) for i in range(len(predictions["predictions"]))]
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions["predictions"],
        mode="lines+markers",
        name="Predicted Commits",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond")
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=predictions["upper"] + predictions["lower"][::-1],
        fill="toself",
        fillcolor="rgba(255, 0, 0, 0.1)",
        line=dict(color="rgba(255, 0, 0, 0)"),
        name="95% Confidence Interval"
    ))
    
    # Layout
    fig.update_layout(
        title=f"Commit Prediction for {repo_name}",
        xaxis_title="Week",
        yaxis_title="Number of Commits",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.title("GitHub Commit Predictor")
    st.markdown("**Neural Network-based prediction of GitHub repository activity**")
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # GitHub Token (optional)
        github_token = st.text_input(
            "GitHub Token (optional)",
            type="password",
            help="Increases API rate limit from 60 to 5000 requests/hour"
        )
        
        # Model checkpoint selection
        st.subheader("Model Checkpoint")
        
        checkpoints = list_checkpoints()
        
        if checkpoints:
            checkpoint_options = ["Latest"] + checkpoints
            selected_checkpoint = st.selectbox(
                "Select Checkpoint",
                options=checkpoint_options,
                index=0
            )
            
            if selected_checkpoint == "Latest":
                latest_path = CHECKPOINT_DIR / "nn_model_latest.pkl"
                if latest_path.exists():
                    checkpoint_path = str(latest_path)
                else:
                    checkpoint_path = checkpoints[0] if checkpoints else None
            else:
                checkpoint_path = selected_checkpoint
        else:
            st.warning("No checkpoints found. Train a model first.")
            checkpoint_path = None
        
        # Prediction horizon
        st.subheader("Prediction Settings")
        horizon = st.slider(
            "Prediction Horizon (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            help="Number of weeks to predict into the future"
        )
        
        st.divider()
        
        # Training link
        st.markdown("### 🎓 Train New Model")
        st.code("python -m src.train_neural_network --list")
    
    # Main content area
    st.header("Repository Prediction")
    
    # GitHub URL input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        github_url = st.text_input(
            "Enter GitHub Repository URL",
            placeholder="https://github.com/owner/repo or owner/repo",
            help="Enter the URL of the GitHub repository to analyze"
        )
    
    with col2:
        predict_button = st.button("Predict", type="primary", use_container_width=True)
    
    # Process prediction
    if predict_button and github_url:
        owner, repo = parse_github_url(github_url)
        
        if not owner or not repo:
            st.error("Invalid GitHub URL. Please use format: https://github.com/owner/repo")
            st.stop()
        
        st.info(f"🔍 Analyzing repository: **{owner}/{repo}**")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch repository info
            status_text.text("📡 Fetching repository information...")
            progress_bar.progress(10)
            
            fetcher = GitHubDataFetcher(github_token)
            repo_info = fetcher.get_repo_info(owner, repo)
            
            st.success(f"Repository found: **{repo_info['full_name']}** {repo_info['stargazers_count']}")
            
            # Step 2: Fetch commits
            status_text.text("📥 Fetching commit history...")
            progress_bar.progress(30)
            
            # Fetch last 2 years of data
            since_date = datetime.now() - timedelta(days=730)
            commits = fetcher.get_commits(owner, repo, since=since_date, max_pages=20)
            
            st.write(f"Fetched **{len(commits)}** commits")
            
            # Step 3: Fetch pull requests
            status_text.text("📥 Fetching pull requests...")
            progress_bar.progress(50)
            
            prs = fetcher.get_pull_requests(owner, repo, max_pages=10)
            st.write(f"Fetched **{len(prs)}** pull requests")
            
            # Step 4: Aggregate data
            status_text.text("Aggregating weekly data...")
            progress_bar.progress(70)
            
            weekly_data = fetcher.aggregate_weekly(commits, prs)
            st.write(f"📅 Aggregated **{len(weekly_data)}** weeks of data")
            
            # Step 5: Load model and predict
            status_text.text("Running neural network inference...")
            progress_bar.progress(85)
            
            if checkpoint_path and Path(checkpoint_path).exists():
                inference = NeuralNetworkInference(checkpoint_path)
                
                predictions = inference.predict(
                    weekly_data["commits"].values,
                    weekly_data["prs_opened"].values if "prs_opened" in weekly_data else None,
                    horizon=horizon
                )
                
                progress_bar.progress(100)
                status_text.empty()
                
                # Display results
                st.header("Prediction Results")
                
                # Prediction plot
                fig = create_prediction_plot(weekly_data, predictions, f"{owner}/{repo}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction table
                st.subheader("📋 Predicted Commits")
                
                last_date = weekly_data["ds"].max()
                pred_df = pd.DataFrame({
                    "Week": [last_date + timedelta(weeks=i+1) for i in range(horizon)],
                    "Predicted Commits": [int(round(p)) for p in predictions["predictions"]],
                    "Lower Bound (95%)": [int(round(p)) for p in predictions["lower"]],
                    "Upper Bound (95%)": [int(round(p)) for p in predictions["upper"]]
                })
                
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Next Week Prediction",
                        f"{int(round(predictions['predictions'][0]))} commits"
                    )
                
                with col2:
                    avg_pred = np.mean(predictions["predictions"])
                    st.metric(
                        f"Avg Next {horizon} Weeks",
                        f"{int(round(avg_pred))} commits/week"
                    )
                
                with col3:
                    historical_avg = weekly_data["commits"].mean()
                    st.metric(
                        "Historical Average",
                        f"{int(round(historical_avg))} commits/week"
                    )
                
                with col4:
                    trend = (avg_pred - historical_avg) / historical_avg * 100 if historical_avg > 0 else 0
                    st.metric(
                        "Predicted Trend",
                        f"{trend:+.1f}%"
                    )
                
            else:
                progress_bar.progress(100)
                status_text.empty()
                
                st.warning("No trained model found. Showing data analysis only.")
                
                # Show historical data visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weekly_data["ds"],
                    y=weekly_data["commits"],
                    mode="lines+markers",
                    name="Weekly Commits",
                    line=dict(color="blue", width=2)
                ))
                
                fig.update_layout(
                    title=f"Historical Commits - {owner}/{repo}",
                    xaxis_title="Week",
                    yaxis_title="Number of Commits",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Train a model with: `python -m src.train_neural_network -r <repository>`")
            
            # Historical data table
            with st.expander("Historical Data"):
                st.dataframe(
                    weekly_data.tail(20).sort_values("ds", ascending=False),
                    use_container_width=True
                )
            
        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Network error: {e}")
            st.info("💡 Try adding a GitHub token to increase API rate limits.")
            
        except ValueError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error: {e}")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error: {e}")
            
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    elif predict_button:
        st.warning("Please enter a GitHub repository URL")
    
    else:
        # Instructions when not running
        st.markdown("""
        ### How to Use
        
        1. **Enter a GitHub Repository URL** in the input field above
           - Examples: `https://github.com/tensorflow/tensorflow` or `facebook/react`
        
        2. **Click "Predict"** to fetch data and run inference
        
        3. **View Predictions** for the upcoming weeks
        
        ### Notes
        
        - The neural network uses historical commit and PR patterns to predict future commits
        - Confidence intervals show prediction uncertainty (wider = more uncertain)
        - For private repositories, you'll need to provide a GitHub token with appropriate permissions
        
        ### 🔧 Training a Custom Model
        
        To train on specific repository data:
        ```bash
        python -m src.train_neural_network -r tensorflow__tensorflow
        ```
        """)
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "GitHub Commit Predictor | Neural Network Inference | "
        "Built with Streamlit & PyTorch"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
