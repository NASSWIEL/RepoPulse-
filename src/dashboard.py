"""
Dashboard - GitHub Activity Predictor (Enhanced v2.0)
=====================================================

Streamlit-based interactive dashboard for visualizing and running
GitHub activity predictions using auto-regressive forecasting.

Features:
- Repository selection dropdown
- Model comparison: Prophet vs FARIMA vs Neural Network
- Configurable forecast horizon
- Interactive Plotly visualizations with confidence intervals
- SMAPE metrics (not MAPE - handles zeros properly)
- Real-time MLflow experiment tracking
- Data validation status display
- A/B testing experiment results
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl import load_repository_data, list_available_repositories, get_repository_summary
from src.model_engine import (
    train_predict_recursive, 
    train_predict_comparative,
    run_full_analysis, 
    ModelType
)

# Import new modules (with fallbacks for missing dependencies)
try:
    from src.neural_network import create_neural_network_wrapper, NeuralNetworkWrapper
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False

try:
    from src.data_validation import create_data_quality_validator
    DATA_VALIDATION_AVAILABLE = True
except ImportError:
    DATA_VALIDATION_AVAILABLE = False

try:
    from src.model_selection import create_auto_model_selector
    MODEL_SELECTION_AVAILABLE = True
except ImportError:
    MODEL_SELECTION_AVAILABLE = False

try:
    from src.ab_testing import create_ab_framework
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="GitHub Activity Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .comparison-table {
        margin-top: 1rem;
    }
    .winner {
        background-color: #d4edda;
        font-weight: bold;
    }
    .confidence-band {
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)


def get_data_directory() -> str:
    """Get the path to the repositories directory."""
    possible_paths = [
        Path("repositories"),
        Path("data/repositories"),
        Path("/app/data/repositories"),
        Path("/app/repositories"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return "repositories"


def train_neural_network_model(df: pd.DataFrame, target_column: str, horizon: int, repo_name: str):
    """Train a neural network model and return results."""
    if not NEURAL_NETWORK_AVAILABLE:
        return None
    
    nn_model = create_neural_network_wrapper()
    
    # Use commits as target, prs_opened as auxiliary feature
    train_df = df.copy()
    
    # Split data
    train_size = len(df) - horizon
    train_data = train_df.iloc[:train_size]
    test_data = train_df.iloc[train_size:]
    
    # Fit model
    nn_model.fit(train_data, target_col=target_column, auxiliary_col='prs_opened')
    
    # Predict with confidence intervals
    predictions = nn_model.predict_with_confidence(horizon, confidence=0.95)
    
    # Calculate metrics
    actual = test_data[target_column].values
    predicted = predictions['yhat'].values[:len(actual)]
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # SMAPE calculation
    smape = 100 * np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8))
    
    return {
        'predictions': predictions,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'smape': smape
        },
        'actual': test_data,
        'has_confidence_intervals': 'yhat_lower' in predictions.columns
    }


def create_forecast_figure_with_ci(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    actual_test: pd.DataFrame,
    target_column: str,
    model_name: str,
    show_ci: bool = True
) -> go.Figure:
    """Create a forecast figure with confidence intervals."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df[target_column],
        mode='lines',
        name='Historical',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Actual test data
    fig.add_trace(go.Scatter(
        x=actual_test['ds'],
        y=actual_test[target_column],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#28A745', width=2),
        marker=dict(size=8)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions['ds'],
        y=predictions['yhat'],
        mode='lines+markers',
        name=f'{model_name} Prediction',
        line=dict(color='#DC3545', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Confidence intervals
    if show_ci and 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['ds'].tolist() + predictions['ds'].tolist()[::-1],
            y=predictions['yhat_upper'].tolist() + predictions['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(220, 53, 69, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'{model_name} Forecast for {target_column.title()}',
        xaxis_title='Date',
        yaxis_title=target_column.replace('_', ' ').title(),
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">GitHub Activity Predictor v2.0</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Auto-regressive forecasting with Prophet, FARIMA & Neural Network comparison"
        "</p>",
        unsafe_allow_html=True
    )
    
    # Feature status indicators
    feature_cols = st.columns(4)
    with feature_cols[0]:
        if NEURAL_NETWORK_AVAILABLE:
            st.success("Neural Network")
        else:
            st.warning("NN: Not installed")
    with feature_cols[1]:
        if DATA_VALIDATION_AVAILABLE:
            st.success("✓ Data Validation")
        else:
            st.warning("✓ Validation: N/A")
    with feature_cols[2]:
        if MODEL_SELECTION_AVAILABLE:
            st.success("Auto Selection")
        else:
            st.warning("Selection: N/A")
    with feature_cols[3]:
        if AB_TESTING_AVAILABLE:
            st.success("A/B Testing")
        else:
            st.warning("A/B: N/A")
        "</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        data_dir = get_data_directory()
        repos = list_available_repositories(data_dir)
        
        if not repos:
            st.error(f"No repositories found in {data_dir}")
            st.stop()
        
        st.success(f"Found {len(repos)} repositories")
        
        # Repository selection
        st.subheader("📁 Repository Selection")
        selected_repo = st.selectbox(
            "Select Repository",
            options=repos,
            index=0,
            help="Choose a GitHub repository to analyze"
        )
        
        # Model configuration
        st.subheader("Model Configuration")
        
        # Mode selection: single model or comparison
        forecast_mode = st.radio(
            "Forecast Mode",
            options=["compare", "single", "auto", "neural"],
            format_func=lambda x: {
                "compare": "🔀 Compare Prophet vs FARIMA",
                "single": "🎯 Single Model",
                "auto": "Auto Select Best Model",
                "neural": "Neural Network (with PRs)"
            }.get(x, x),
            index=0,
            help="Compare models, run single model, auto-select best, or use neural network"
        )
        
        if forecast_mode == "single":
            model_type: ModelType = st.selectbox(
                "Model Type",
                options=["prophet", "farima"],
                index=0,
                format_func=lambda x: x.upper(),
                help="Prophet: Seasonal decomposition\nFARIMA: Long memory time series"
            )
        else:
            model_type = None  # Will run both or auto-select
        
        # Confidence intervals option
        show_confidence = st.checkbox(
            "Show Confidence Intervals",
            value=True,
            help="Display prediction uncertainty bands (95% CI)"
        )
        
        # Target metric selection
        target_metric = st.selectbox(
            "Target Metric",
            options=["commits", "new_stars", "issues_opened", "prs_opened"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
            help="Select which metric to forecast"
        )
        
        # Forecast horizon
        horizon = st.slider(
            "Forecast Horizon (weeks)",
            min_value=2,
            max_value=12,
            value=8,
            help="Number of weeks to forecast into the future"
        )
        
        st.divider()
        
        # MLflow info
        st.subheader("MLflow Tracking")
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        st.info(f"Tracking URI: {mlflow_uri}")
        
        # Run button
        st.divider()
        run_button = st.button(
            "Lancer l'entraînement",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    repo_path = os.path.join(data_dir, selected_repo)
    
    # Display repository info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"{selected_repo.replace('__', '/')}")
    
    # Load and display data summary
    with st.spinner("Loading repository data..."):
        try:
            df = load_repository_data(repo_path)
            summary = get_repository_summary(repo_path)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Display summary metrics
    with col2:
        st.metric("Total Weeks", len(df))
    
    # Data Validation Section
    if DATA_VALIDATION_AVAILABLE:
        with st.expander("Data Quality Validation", expanded=False):
            try:
                validator = create_data_quality_validator()
                report = validator.validate(df)
                
                val_cols = st.columns(4)
                with val_cols[0]:
                    if report.is_valid:
                        st.success("✓ Schema Valid")
                    else:
                        st.error("✗ Schema Invalid")
                with val_cols[1]:
                    st.metric("Anomalies", len(report.anomalies))
                with val_cols[2]:
                    st.metric("Drift Issues", len(report.drift_issues))
                with val_cols[3]:
                    st.metric("Warnings", len(report.warnings))
                
                if report.warnings:
                    st.warning("Warnings: " + "; ".join(report.warnings[:3]))
                if report.anomalies:
                    st.info(f"Detected {len(report.anomalies)} anomalies in the data")
            except Exception as e:
                st.warning(f"Could not validate data: {e}")
    
    # Data overview section
    st.subheader("Data Overview")
    
    metric_cols = st.columns(4)
    metrics = ["commits", "new_stars", "issues_opened", "prs_opened"]
    metric_labels = ["Commits", "New Stars", "Issues Opened", "PRs Opened"]
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        with metric_cols[i]:
            total = int(df[metric].sum())
            avg = df[metric].mean()
            st.metric(
                label=label,
                value=f"{total:,}",
                delta=f"~{avg:.1f}/week"
            )
    
    # Time series preview
    with st.expander("Historical Data Preview", expanded=True):
        fig_overview = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = ["#2E86AB", "#F6AE2D", "#28A745", "#DC3545"]
        
        for (row, col), metric, color in zip(positions, metrics, colors):
            fig_overview.add_trace(
                go.Scatter(
                    x=df["ds"],
                    y=df[metric],
                    mode="lines",
                    name=metric.replace("_", " ").title(),
                    line=dict(color=color, width=1.5),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_overview.update_layout(
            height=500,
            template="plotly_white",
            title_text="Weekly Activity Overview"
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
    
    # Data table preview
    with st.expander("📋 Raw Data Table"):
        st.dataframe(
            df.head(50).style.format({
                "commits": "{:,.0f}",
                "new_stars": "{:,.0f}",
                "issues_opened": "{:,.0f}",
                "prs_opened": "{:,.0f}"
            }),
            use_container_width=True
        )
    
    st.divider()
    
    # Forecasting section
    if run_button:
        st.header("Recursive Forecast Results")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if forecast_mode == "compare":
                # COMPARATIVE MODE: Prophet vs FARIMA
                status_text.text("Training Prophet model...")
                progress_bar.progress(20)
                
                results = train_predict_comparative(
                    df=df,
                    target_column=target_metric,
                    horizon=horizon,
                    repo_name=selected_repo
                )
                
                progress_bar.progress(80)
                status_text.text("Generating comparison visualization...")
                
                progress_bar.progress(100)
                status_text.empty()
                
                st.success("Comparative training complete!")
                
                # ========== METRICS COMPARISON TABLE ==========
                st.subheader("📏 Model Performance Comparison")
                
                # Create comparison dataframe
                prophet_m = results["prophet_metrics"]
                farima_m = results["farima_metrics"]
                
                comparison_data = {
                    "Metric": ["RMSE ↓", "MAE ↓", "SMAPE ↓"],
                    "Prophet": [
                        f"{prophet_m['rmse']:.4f}",
                        f"{prophet_m['mae']:.4f}",
                        f"{prophet_m['smape']:.2f}%"
                    ],
                    "FARIMA": [
                        f"{farima_m['rmse']:.4f}",
                        f"{farima_m['mae']:.4f}",
                        f"{farima_m['smape']:.2f}%"
                    ],
                    "Winner": []
                }
                
                # Determine winners
                for metric_key in ["rmse", "mae", "smape"]:
                    if prophet_m[metric_key] < farima_m[metric_key]:
                        comparison_data["Winner"].append("🏆 Prophet")
                    elif farima_m[metric_key] < prophet_m[metric_key]:
                        comparison_data["Winner"].append("🏆 FARIMA")
                    else:
                        comparison_data["Winner"].append("Tie")
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display as columns with metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Prophet")
                    st.metric("RMSE", f"{prophet_m['rmse']:.4f}")
                    st.metric("MAE", f"{prophet_m['mae']:.4f}")
                    st.metric("SMAPE", f"{prophet_m['smape']:.2f}%")
                
                with col2:
                    st.markdown("### FARIMA")
                    st.metric("RMSE", f"{farima_m['rmse']:.4f}")
                    st.metric("MAE", f"{farima_m['mae']:.4f}")
                    st.metric("SMAPE", f"{farima_m['smape']:.2f}%")
                
                with col3:
                    st.markdown("### Winner")
                    # Overall winner
                    prophet_wins = sum(1 for w in comparison_data["Winner"] if "Prophet" in w)
                    farima_wins = sum(1 for w in comparison_data["Winner"] if "FARIMA" in w)
                    
                    if prophet_wins > farima_wins:
                        st.success("🏆 **Prophet** wins overall!")
                    elif farima_wins > prophet_wins:
                        st.success("🏆 **FARIMA** wins overall!")
                    else:
                        st.info("**Tie** - Both models perform similarly")
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # ========== COMPARATIVE PLOT ==========
                st.subheader("Forecast Visualization: Prophet vs FARIMA")
                st.plotly_chart(results["comparison_figure"], use_container_width=True)
                
                # ========== PREDICTIONS TABLES ==========
                st.subheader("📋 Prediction Details")
                
                tab1, tab2 = st.tabs(["Prophet Predictions", "FARIMA Predictions"])
                
                with tab1:
                    prophet_pred = results["prophet_predictions"].copy()
                    prophet_pred["ds"] = prophet_pred["ds"].dt.strftime("%Y-%m-%d")
                    prophet_pred = prophet_pred.rename(columns={
                        "ds": "Date", "y_pred": "Predicted", "y_actual": "Actual", "step": "Step"
                    })
                    prophet_pred["Error"] = prophet_pred["Predicted"] - prophet_pred["Actual"]
                    
                    st.dataframe(
                        prophet_pred.style.format({
                            "Predicted": "{:.2f}",
                            "Actual": "{:.0f}",
                            "Error": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                
                with tab2:
                    farima_pred = results["farima_predictions"].copy()
                    farima_pred["ds"] = farima_pred["ds"].dt.strftime("%Y-%m-%d")
                    farima_pred = farima_pred.rename(columns={
                        "ds": "Date", "y_pred": "Predicted", "y_actual": "Actual", "step": "Step"
                    })
                    farima_pred["Error"] = farima_pred["Predicted"] - farima_pred["Actual"]
                    
                    st.dataframe(
                        farima_pred.style.format({
                            "Predicted": "{:.2f}",
                            "Actual": "{:.0f}",
                            "Error": "{:.2f}"
                        }),
                        use_container_width=True
                    )
            
            elif forecast_mode == "neural":
                # NEURAL NETWORK MODE
                if not NEURAL_NETWORK_AVAILABLE:
                    st.error("Neural Network module not available. Install with: pip install torch")
                    st.stop()
                
                status_text.text("Training Neural Network model...")
                progress_bar.progress(30)
                
                nn_results = train_neural_network_model(df, target_metric, horizon, selected_repo)
                
                progress_bar.progress(80)
                
                if nn_results is None:
                    st.error("Failed to train neural network model")
                else:
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    st.success("Neural Network training complete!")
                    
                    # Metrics
                    st.subheader("📏 Neural Network Performance")
                    nn_m = nn_results['metrics']
                    
                    m_cols = st.columns(3)
                    with m_cols[0]:
                        st.metric("RMSE", f"{nn_m['rmse']:.4f}")
                    with m_cols[1]:
                        st.metric("MAE", f"{nn_m['mae']:.4f}")
                    with m_cols[2]:
                        st.metric("SMAPE", f"{nn_m['smape']:.2f}%")
                    
                    # Plot with confidence intervals
                    st.subheader("Neural Network Forecast with Confidence Intervals")
                    
                    train_size = len(df) - horizon
                    train_data = df.iloc[:train_size]
                    test_data = df.iloc[train_size:]
                    
                    fig_nn = create_forecast_figure_with_ci(
                        train_data,
                        nn_results['predictions'],
                        test_data,
                        target_metric,
                        "Neural Network",
                        show_ci=show_confidence
                    )
                    st.plotly_chart(fig_nn, use_container_width=True)
                    
                    st.info("The Neural Network model exploits the correlation between Pull Requests and Commits for improved prediction accuracy.")
            
            elif forecast_mode == "auto":
                # AUTO-SELECT BEST MODEL
                if not MODEL_SELECTION_AVAILABLE:
                    st.error("Model Selection module not available.")
                    st.stop()
                
                status_text.text("Running automatic model selection...")
                progress_bar.progress(20)
                
                selector = create_auto_model_selector()
                
                progress_bar.progress(40)
                status_text.text("Evaluating Prophet, FARIMA, and Neural Network...")
                
                try:
                    best_model, selection_results = selector.select_best_model(
                        df, target_column=target_metric
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    st.success(f"Auto-selection complete! Best model: **{selection_results['best_model_type'].upper()}**")
                    
                    # Display all model scores
                    st.subheader("📏 Model Comparison (Auto-Selection)")
                    
                    scores_df = pd.DataFrame([
                        {
                            "Model": mt.upper(),
                            "SMAPE": f"{score:.2f}%",
                            "Selected": "✓" if mt == selection_results['best_model_type'] else ""
                        }
                        for mt, score in selection_results['all_scores'].items()
                    ])
                    
                    st.dataframe(scores_df, use_container_width=True, hide_index=True)
                    
                    # Generate predictions with best model
                    st.subheader(f"Forecast using {selection_results['best_model_type'].upper()}")
                    
                    train_size = len(df) - horizon
                    train_data = df.iloc[:train_size]
                    test_data = df.iloc[train_size:]
                    
                    best_model.fit(train_data, target_col=target_metric)
                    predictions = best_model.predict(horizon)
                    
                    # Create figure
                    fig_auto = create_forecast_figure_with_ci(
                        train_data,
                        predictions,
                        test_data,
                        target_metric,
                        selection_results['best_model_type'].upper(),
                        show_ci=show_confidence and 'yhat_lower' in predictions.columns
                    )
                    st.plotly_chart(fig_auto, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Auto-selection failed: {e}")
                
            else:
                # SINGLE MODEL MODE
                status_text.text(f"Training {model_type.upper()} model...")
                progress_bar.progress(30)
                
                results = train_predict_recursive(
                    df=df,
                    target_column=target_metric,
                    model_type=model_type,
                    horizon=horizon,
                    repo_name=selected_repo
                )
                
                progress_bar.progress(90)
                status_text.text("Generating visualization...")
                
                progress_bar.progress(100)
                status_text.empty()
                
                if results.get("mlflow_run_id"):
                    st.success(f"Training complete! MLflow Run ID: `{results['mlflow_run_id']}`")
                else:
                    st.success("Training complete!")
                
                # Metrics display
                st.subheader("📏 Performance Metrics")
                
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.metric(
                        label="RMSE",
                        value=f"{results['metrics']['rmse']:.4f}",
                        help="Root Mean Squared Error - Lower is better"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="MAE",
                        value=f"{results['metrics']['mae']:.4f}",
                        help="Mean Absolute Error - Lower is better"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        label="SMAPE",
                        value=f"{results['metrics']['smape']:.2f}%",
                        help="Symmetric Mean Absolute Percentage Error - Handles zeros properly"
                    )
                
                # Main forecast plot
                st.subheader("Forecast Visualization")
                st.plotly_chart(results["figure"], use_container_width=True)
                
                # Predictions table
                st.subheader("📋 Prediction Details")
                
                pred_df = results["predictions"].copy()
                pred_df["ds"] = pred_df["ds"].dt.strftime("%Y-%m-%d")
                pred_df = pred_df.rename(columns={
                    "ds": "Date", "y_pred": "Predicted", "y_actual": "Actual", "step": "Step"
                })
                pred_df["Error"] = pred_df["Predicted"] - pred_df["Actual"]
                
                st.dataframe(
                    pred_df.style.format({
                        "Predicted": "{:.2f}",
                        "Actual": "{:.0f}",
                        "Error": "{:.2f}"
                    }),
                    use_container_width=True
                )
            
            # Explanation of methodology
            with st.expander("About the Forecasting Methodology"):
                st.markdown("""
                ### Recursive Multi-Step Forecasting (No Data Leakage)
                
                This implementation ensures **strict data isolation** - the model never sees future data:
                
                1. **Split**: Data is split into training (historical) and test (horizon) periods
                2. **Train**: Model is trained ONLY on data available up to current time point
                3. **Predict**: Forecast only the next week (t+1)
                4. **Feedback**: Add the **prediction** (not actual) to training data
                5. **Repeat**: Continue until horizon is reached
                
                ### Models
                
                **Prophet** (Facebook):
                - Additive decomposition model
                - Handles seasonality (yearly, weekly)
                - Good for data with strong seasonal patterns
                
                **FARIMA** (Fractional ARIMA):
                - Fractional differencing for long memory
                - Captures persistence in time series
                - Better for data with long-range dependencies
                
                ### 📏 Metrics
                
                - **RMSE**: Root Mean Squared Error (penalizes large errors)
                - **MAE**: Mean Absolute Error (average error magnitude)
                - **SMAPE**: Symmetric MAPE (handles zeros, bounded 0-200%)
                """)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during training: {e}")
            
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # Instructions when not running
        st.info(
            "👈 Configure the model parameters in the sidebar and click "
            "**'Lancer l'entraînement'** to start the recursive forecasting."
        )
        
        # Show what will be done
        st.subheader("🎯 Forecast Configuration Summary")
        
        config_cols = st.columns(4)
        with config_cols[0]:
            st.info(f"**Repository:**\n{selected_repo}")
        with config_cols[1]:
            if forecast_mode == "compare":
                st.info("**Mode:**\n🔀 Prophet vs FARIMA")
            else:
                st.info(f"**Model:**\n{model_type.upper()}")
        with config_cols[2]:
            st.info(f"**Target:**\n{target_metric.replace('_', ' ').title()}")
        with config_cols[3]:
            st.info(f"**Horizon:**\n{horizon} weeks")
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "GitHub Activity Predictor v2.0 | Enhanced MLOps Pipeline | "
        "Built with Streamlit, Prophet, FARIMA, Neural Networks, FastAPI, Prefect & MLflow"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
