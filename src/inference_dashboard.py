"""
Inference Dashboard - GitHub Activity Predictor
================================================

Streamlit dashboard with:
- GitHub URL input for real-time prediction
- Automatic data extraction and aggregation
- Three model choices: Prophet, FARIMA, Neural Network
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
    page_title="GitHub Commit Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
CHECKPOINT_DIR = Path("checkpoints")


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

    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            self.headers["Authorization"] = f"token {token}"

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

    def get_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_pages: int = 5
    ) -> List[Dict]:
        """Fetch issues (excluding pull requests)."""
        issues = []
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {"per_page": 100, "state": state}

        for page in range(1, max_pages + 1):
            params["page"] = page
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                break
            page_issues = response.json()
            if not page_issues:
                break
            # GitHub API returns PRs as issues too; filter them out
            issues.extend([i for i in page_issues if "pull_request" not in i])

        return issues

    def aggregate_weekly(self, commits: List[Dict], prs: List[Dict]) -> pd.DataFrame:
        """
        Aggregate data into weekly time series.
        """
        commit_dates = []
        for c in commits:
            try:
                date_str = c.get("commit", {}).get("author", {}).get("date")
                if date_str:
                    commit_dates.append(pd.to_datetime(date_str))
            except:
                pass

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

        commit_df = pd.DataFrame({"date": commit_dates})
        commit_df["date"] = pd.to_datetime(commit_df["date"]).dt.tz_localize(None)
        commit_df.set_index("date", inplace=True)
        weekly_commits = commit_df.resample("W").size()

        if pr_dates:
            pr_df = pd.DataFrame({"date": pr_dates})
            pr_df["date"] = pd.to_datetime(pr_df["date"]).dt.tz_localize(None)
            pr_df.set_index("date", inplace=True)
            weekly_prs = pr_df.resample("W").size()
        else:
            weekly_prs = pd.Series(0, index=weekly_commits.index)

        result = pd.DataFrame({
            "ds": weekly_commits.index,
            "commits": weekly_commits.values,
            "prs_opened": weekly_prs.reindex(weekly_commits.index, fill_value=0).values
        })

        result = result.sort_values("ds").reset_index(drop=True)
        return result


# ============================================================
# Neural Network Inference (sklearn_ensemble + torch support)
# ============================================================

def compute_features_for_inference(data: np.ndarray, idx: int, lookback: int) -> np.ndarray:
    """
    Compute the same enhanced features that train_neural_network uses.
    Must stay in sync with _train_fallback_with_tracking.compute_features.
    """
    window = data[idx:idx + lookback]
    features = list(window)

    features.extend([
        np.mean(window),
        np.std(window) + 1e-8,
        np.min(window),
        np.max(window),
        np.median(window),
        window[-1] - window[0],
    ])

    if lookback >= 4:
        features.append(np.mean(window[-4:]))
        features.append(np.mean(window[:4]))
    if lookback >= 8:
        features.append(np.mean(window[-8:]))
    if lookback >= 12:
        features.append(np.mean(window[-12:]))

    if lookback >= 2:
        features.append(window[-1] - window[-2])
    if lookback >= 4:
        features.append(window[-1] - window[-4])
        features.append(np.mean(window[-2:]) - np.mean(window[-4:-2]))
    if lookback >= 8:
        features.append(window[-1] - window[-8])

    if lookback >= 4:
        features.append(np.std(window[-4:]) / (np.mean(window[-4:]) + 1e-8))
    if lookback >= 8:
        features.append(np.std(window[-8:]) / (np.mean(window[-8:]) + 1e-8))

    features.append(np.percentile(window, 25))
    features.append(np.percentile(window, 75))

    x = np.arange(len(window))
    if len(window) > 1:
        slope = np.polyfit(x, window, 1)[0]
        features.append(slope)
    else:
        features.append(0)

    return np.array(features)


class NeuralNetworkInference:
    """Performs inference using saved checkpoint (sklearn_ensemble or torch)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model = None
        self.use_torch = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from file, handling both sklearn_ensemble and torch."""
        with open(checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)

        model_state = self.checkpoint["model_state"]
        model_type = model_state.get("type", "")

        if model_type in ("sklearn", "sklearn_ensemble"):
            # sklearn-based model (VotingRegressor / single estimator)
            self.model = model_state["model"]
            self.scaler = model_state["scaler"]
            self.feature_scaler = model_state.get("feature_scaler", None)
            self.use_torch = False

        elif "model_architecture" in model_state:
            # PyTorch model
            if not check_torch_available():
                raise RuntimeError("PyTorch required for this checkpoint")
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

            self.scaler = model_state.get("scalers", {}).get("commits")
            self.feature_scaler = None
            self.use_torch = True

        else:
            raise ValueError(f"Unknown checkpoint type: {model_type}")

        self.lookback = model_state["lookback_window"]
        self.use_prs = model_state.get("use_prs",
                                       model_state.get("config", {}).get("use_prs_correlation", False))

    def predict(
        self,
        commits: np.ndarray,
        prs: Optional[np.ndarray] = None,
        horizon: int = 4
    ) -> Dict[str, Any]:
        """Make multi-step recursive predictions."""
        if len(commits) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} weeks of history")

        predictions = []
        lower_bounds = []
        upper_bounds = []

        # Working copy of recent history
        current_commits = list(commits.copy())

        for step in range(horizon):
            # Normalise the lookback window
            window_raw = np.array(current_commits[-self.lookback:]).reshape(-1, 1)
            window_norm = self.scaler.transform(window_raw).flatten()

            if not self.use_torch:
                # sklearn_ensemble path: compute enhanced features
                features = compute_features_for_inference(
                    window_norm, idx=0, lookback=self.lookback
                )
                X = features.reshape(1, -1)
                if self.feature_scaler is not None:
                    X = self.feature_scaler.transform(X)

                pred_norm = self.model.predict(X)[0]
            else:
                # torch path: raw lookback window
                import torch
                X = window_norm.reshape(1, -1)
                with torch.no_grad():
                    pred_norm = self.model(torch.FloatTensor(X)).item()

            # Inverse transform
            pred = self.scaler.inverse_transform(np.array([[pred_norm]]))[0, 0]
            pred = max(0, pred)
            predictions.append(pred)

            # Confidence interval
            std_estimate = np.std(commits[-self.lookback:]) * (1 + 0.1 * step)
            lower_bounds.append(max(0, pred - 1.96 * std_estimate))
            upper_bounds.append(pred + 1.96 * std_estimate)

            # Slide window forward
            current_commits.append(pred)

        return {
            "predictions": predictions,
            "lower": lower_bounds,
            "upper": upper_bounds
        }


# ============================================================
# Prophet / FARIMA wrappers (lightweight, run directly)
# ============================================================

def run_prophet_forecast(weekly_data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """Run Prophet model on weekly data."""
    from prophet import Prophet
    import warnings
    warnings.filterwarnings("ignore")

    df = weekly_data[["ds", "commits"]].copy()
    df = df.rename(columns={"commits": "y"})
    df = df.sort_values("ds").reset_index(drop=True)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    working_train = df.copy()

    for step in range(horizon):
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive"
        )
        m.fit(working_train)

        future = m.make_future_dataframe(periods=1, freq="W")
        forecast = m.predict(future)

        pred = max(0, float(forecast["yhat"].iloc[-1]))
        lo = max(0, float(forecast["yhat_lower"].iloc[-1]))
        hi = float(forecast["yhat_upper"].iloc[-1])

        predictions.append(pred)
        lower_bounds.append(lo)
        upper_bounds.append(hi)

        new_row = pd.DataFrame({"ds": [forecast["ds"].iloc[-1]], "y": [pred]})
        working_train = pd.concat([working_train, new_row], ignore_index=True)

    return {"predictions": predictions, "lower": lower_bounds, "upper": upper_bounds}


def run_farima_forecast(weekly_data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """Run FARIMA model on weekly data."""
    from src.model_engine import FARIMAWrapper
    import warnings
    warnings.filterwarnings("ignore")

    df = weekly_data[["ds", "commits"]].copy()
    df = df.rename(columns={"commits": "y"})
    df = df.sort_values("ds").reset_index(drop=True)

    working_train = df.copy()

    predictions = []
    lower_bounds = []
    upper_bounds = []

    for step in range(horizon):
        model = FARIMAWrapper(d=0.4, arima_order=(2, 0, 2))
        model.fit(working_train)
        pred = max(0, model.predict(periods=1))
        predictions.append(pred)

        std_estimate = np.std(working_train["y"].values[-12:]) * (1 + 0.1 * step)
        lower_bounds.append(max(0, pred - 1.96 * std_estimate))
        upper_bounds.append(pred + 1.96 * std_estimate)

        next_date = working_train["ds"].max() + timedelta(weeks=1)
        new_row = pd.DataFrame({"ds": [next_date], "y": [pred]})
        working_train = pd.concat([working_train, new_row], ignore_index=True)

    return {"predictions": predictions, "lower": lower_bounds, "upper": upper_bounds}


# ============================================================
# Visualization
# ============================================================

def create_prediction_plot(
    historical_data: pd.DataFrame,
    predictions: Dict[str, Any],
    repo_name: str,
    model_label: str = "Model"
) -> go.Figure:
    """Create an interactive prediction plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_data["ds"],
        y=historical_data["commits"],
        mode="lines+markers",
        name="Historical Commits",
        line=dict(color="blue", width=2),
        marker=dict(size=4)
    ))

    last_date = historical_data["ds"].max()
    last_commit_value = float(historical_data["commits"].iloc[-1])

    # Build prediction series that starts from the last historical point
    # so the red line is visually connected to the blue line
    bridge_dates = [last_date] + [last_date + timedelta(weeks=i + 1)
                                   for i in range(len(predictions["predictions"]))]
    bridge_values = [last_commit_value] + list(predictions["predictions"])

    pred_dates = bridge_dates[1:]  # pure prediction dates (no bridge)

    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_values,
        mode="lines+markers",
        name=f"Predicted ({model_label})",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond")
    ))

    fig.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=predictions["upper"] + predictions["lower"][::-1],
        fill="toself",
        fillcolor="rgba(255, 0, 0, 0.1)",
        line=dict(color="rgba(255, 0, 0, 0)"),
        name="95% Confidence Interval"
    ))

    fig.update_layout(
        title=f"Commit Prediction for {repo_name} ({model_label})",
        xaxis_title="Week",
        yaxis_title="Number of Commits",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


# ============================================================
# Main application
# ============================================================

def main():
    """Main dashboard application."""

    st.title("GitHub Commit Predictor")
    st.markdown("**Prediction of GitHub repository commit activity**")

    st.divider()

    # ------ Sidebar ------
    with st.sidebar:
        st.header("Configuration")

        # Model choice
        st.subheader("Model")
        model_choice = st.radio(
            "Select Model",
            options=["neural_network", "prophet", "farima"],
            format_func=lambda x: {
                "neural_network": "Neural Network (our model)",
                "prophet": "Prophet",
                "farima": "FARIMA"
            }[x],
            index=0
        )

        # Prediction horizon
        st.subheader("Prediction Settings")
        horizon = st.slider(
            "Prediction Horizon (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            help="Number of weeks to predict into the future"
        )

    # ------ Resolve latest checkpoint for NN ------
    checkpoint_path = None
    latest_path = CHECKPOINT_DIR / "nn_model_latest.pkl"
    if latest_path.exists():
        checkpoint_path = str(latest_path)

    # ------ Main content ------
    st.header("Repository Prediction")

    col1, col2 = st.columns([3, 1])
    with col1:
        github_url = st.text_input(
            "Enter GitHub Repository URL",
            placeholder="https://github.com/owner/repo or owner/repo",
            help="Enter the URL of the GitHub repository to analyze"
        )
    with col2:
        predict_button = st.button("Predict", type="primary", use_container_width=True)

    # ------ Process prediction ------
    if predict_button and github_url:
        owner, repo = parse_github_url(github_url)

        if not owner or not repo:
            st.error("Invalid GitHub URL. Please use format: https://github.com/owner/repo")
            st.stop()

        st.info(f"Analyzing repository: **{owner}/{repo}**")

        progress_bar = st.progress(0)
        steps_container = st.container()

        # Pipeline step definitions: (label, progress_pct)
        STEPS = [
            ("Fetching repository information", 10),
            ("Fetching commits", 30),
            ("Fetching pull requests", 45),
            ("Fetching issues", 55),
            ("Ingesting and aggregating data", 70),
            ("Running prediction", 85),
        ]

        # Track status: 0=pending, 1=running, 2=completed
        step_status = [0] * len(STEPS)

        def update_progress(step_idx: int, status: str):
            """Update the status of a specific step and render only up to current step."""
            _, pct = STEPS[step_idx]
            progress_bar.progress(pct)
            
            if status == "running":
                step_status[step_idx] = 1
            elif status == "completed":
                step_status[step_idx] = 2
            
            # Build the display for steps up to and including the current step
            lines = []
            for i in range(step_idx + 1):
                step_name = STEPS[i][0]
                if step_status[i] == 2:
                    lines.append(f"{i + 1}/{len(STEPS)} {step_name} — **Completed**")
                elif step_status[i] == 1:
                    lines.append(f"{i + 1}/{len(STEPS)} {step_name} — *Running*")
            
            steps_container.markdown("  \n".join(lines))

        try:
            fetcher = GitHubDataFetcher()

            # Step 1 - repository info
            update_progress(0, "running")
            repo_info = fetcher.get_repo_info(owner, repo)
            update_progress(0, "completed")

            # Step 2 - commits
            update_progress(1, "running")
            since_date = datetime.now() - timedelta(days=730)
            commits = fetcher.get_commits(owner, repo, since=since_date, max_pages=20)
            update_progress(1, "completed")

            # Step 3 - pull requests
            update_progress(2, "running")
            prs = fetcher.get_pull_requests(owner, repo, max_pages=10)
            update_progress(2, "completed")

            # Step 4 - issues
            update_progress(3, "running")
            issues = fetcher.get_issues(owner, repo, max_pages=5)
            update_progress(3, "completed")

            # Step 5 - aggregate / ingest
            update_progress(4, "running")
            weekly_data = fetcher.aggregate_weekly(commits, prs)
            update_progress(4, "completed")

            # -- One-line metrics summary table --
            metrics_df = pd.DataFrame([{
                "Commits": len(commits),
                "Pull Requests": len(prs),
                "Issues": len(issues),
                "Stars": repo_info.get("stargazers_count", 0)
            }])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Step 6 - model inference
            active_model = model_choice
            model_label = {
                "neural_network": "Neural Network",
                "prophet": "Prophet",
                "farima": "FARIMA"
            }[active_model]

            update_progress(5, "running")

            if active_model == "neural_network":
                if checkpoint_path:
                    inference = NeuralNetworkInference(checkpoint_path)
                    predictions = inference.predict(
                        weekly_data["commits"].values,
                        weekly_data["prs_opened"].values if "prs_opened" in weekly_data else None,
                        horizon=horizon
                    )
                else:
                    st.warning("No trained neural network checkpoint found. Falling back to Prophet.")
                    active_model = "prophet"
                    model_label = "Prophet"
                    predictions = run_prophet_forecast(weekly_data, horizon)
            elif active_model == "prophet":
                predictions = run_prophet_forecast(weekly_data, horizon)
            else:  # farima
                predictions = run_farima_forecast(weekly_data, horizon)

            update_progress(5, "completed")
            progress_bar.progress(100)

            # ------ Display results ------
            st.header("Prediction Results")

            fig = create_prediction_plot(weekly_data, predictions, f"{owner}/{repo}", model_label)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Predicted Commits")
            last_date = weekly_data["ds"].max()
            pred_df = pd.DataFrame({
                "Week": [last_date + timedelta(weeks=i + 1) for i in range(horizon)],
                "Predicted Commits": [int(round(p)) for p in predictions["predictions"]],
                "Lower Bound (95%)": [int(round(p)) for p in predictions["lower"]],
                "Upper Bound (95%)": [int(round(p)) for p in predictions["upper"]]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Next Week Prediction", f"{int(round(predictions['predictions'][0]))} commits")
            with col2:
                avg_pred = np.mean(predictions["predictions"])
                st.metric(f"Avg Next {horizon} Weeks", f"{int(round(avg_pred))} commits/week")
            with col3:
                historical_avg = weekly_data["commits"].mean()
                st.metric("Historical Average", f"{int(round(historical_avg))} commits/week")
            with col4:
                trend = (avg_pred - historical_avg) / historical_avg * 100 if historical_avg > 0 else 0
                st.metric("Predicted Trend", f"{trend:+.1f}%")

            # Expandable raw data
            with st.expander("Historical Data"):
                st.dataframe(
                    weekly_data.tail(20).sort_values("ds", ascending=False),
                    use_container_width=True
                )

        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Network error: {e}")

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
        st.markdown("""
        ### How to Use

        1. Enter a GitHub Repository URL in the input field above
           - Examples: `https://github.com/tensorflow/tensorflow` or `facebook/react`

        2. Select a model from the sidebar (Neural Network, Prophet, or FARIMA)

        3. Click **Predict** to fetch data and run inference

        4. View predictions for the upcoming weeks
        """)

    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "GitHub Commit Predictor | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
