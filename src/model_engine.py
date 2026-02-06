"""
Model Engine - GitHub Activity Predictor (Refactored)
=====================================================

This module implements auto-regressive (recursive) multi-step forecasting
for predicting GitHub repository activity.

Key Features:
- Recursive Multi-step Forecasting with STRICT data isolation (no leakage)
- Support for Prophet and FARIMA (Fractional ARIMA) models
- SMAPE metric instead of MAPE (handles zeros properly)
- MLflow experiment tracking
- Comparative visualization (Prophet vs FARIMA)

Algorithm (Recursive Loop - No Data Leakage):
1. Split data into Train (historical) / Test (horizon h weeks)
2. For each time step t from 1 to h:
   - Train model ONLY on data from start to current position (NO future data)
   - Predict only t+1
   - Add predicted value ŷ_{t+1} to working history
   - Repeat until t+h
3. Calculate metrics (RMSE, MAE, SMAPE) on test period
4. Log everything to MLflow
"""

import os
import logging
from typing import Tuple, Dict, Optional, Literal, List, Union
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model types
ModelType = Literal["prophet", "farima"]


# =============================================================================
# METRICS
# =============================================================================

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Σ |y_pred - y_true| / ((|y_true| + |y_pred|) / 2)
    
    Handles the case where both y_true and y_pred are 0 (returns 0, not NaN).
    
    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        
    Returns:
        SMAPE value as percentage (0-200 scale, typically 0-100 for good models).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Handle case where both are 0: set ratio to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            (y_true == 0) & (y_pred == 0),
            0.0,
            numerator / np.where(denominator == 0, 1, denominator)
        )
    
    smape = 100 * np.mean(ratio)
    return float(smape)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all forecasting metrics.
    
    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        
    Returns:
        Dictionary with rmse, mae, smape.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape
    }


# =============================================================================
# PROPHET WRAPPER
# =============================================================================

class ProphetWrapper:
    """Wrapper for Facebook Prophet model with strict data isolation."""
    
    def __init__(self, **kwargs):
        """Initialize Prophet with custom parameters."""
        self.model = None
        self.params = {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "additive",  # Changed to additive for count data
            **kwargs
        }
        self._last_date = None
    
    def fit(self, df: pd.DataFrame) -> "ProphetWrapper":
        """
        Fit Prophet model on ONLY the provided data (no future data).
        
        Args:
            df: DataFrame with columns 'ds' (datetime) and 'y' (value).
                MUST NOT contain any future data points.
            
        Returns:
            Self for chaining.
        """
        from prophet import Prophet
        
        # Prophet requires specific column names - create a clean copy
        train_df = df[["ds", "y"]].copy()
        train_df = train_df.sort_values("ds").reset_index(drop=True)
        
        # Store last date for prediction reference
        self._last_date = train_df["ds"].iloc[-1]
        
        # Suppress Prophet's verbose output
        self.model = Prophet(**self.params)
        
        import logging as py_logging
        cmdstanpy_logger = py_logging.getLogger("cmdstanpy")
        prophet_logger = py_logging.getLogger("prophet")
        cmdstanpy_logger.setLevel(py_logging.WARNING)
        prophet_logger.setLevel(py_logging.WARNING)
        
        self.model.fit(train_df)
        return self
    
    def predict(self, periods: int = 1) -> float:
        """
        Predict the next value(s).
        
        Args:
            periods: Number of periods to predict (default: 1).
            
        Returns:
            Predicted value for the next period.
        """
        # Create future dataframe starting from last training date
        future = self.model.make_future_dataframe(periods=periods, freq="W")
        forecast = self.model.predict(future)
        
        # Return only the last predicted value (the true forecast, not fitted values)
        return float(forecast.iloc[-1]["yhat"])


# =============================================================================
# FARIMA WRAPPER (Fractional ARIMA with Long Memory)
# =============================================================================

class FARIMAWrapper:
    """
    Wrapper for FARIMA (Fractional ARIMA) model.
    
    FARIMA extends ARIMA by allowing fractional differencing parameter d,
    which captures long-memory dependencies in time series data.
    
    Algorithm:
    1. Apply fractional differencing with parameter d (0 < d < 0.5 for stationarity)
    2. Fit ARIMA on the differenced series
    3. For prediction, forecast with ARIMA then apply fractional integration
    """
    
    def __init__(
        self, 
        d: float = 0.4,
        arima_order: Tuple[int, int, int] = (2, 0, 2),
        **kwargs
    ):
        """
        Initialize FARIMA model.
        
        Args:
            d: Fractional differencing parameter (0 < d < 0.5 for long memory).
            arima_order: (p, d_int, q) order for the ARIMA on differenced series.
                        Note: d_int is typically 0 since we do fractional differencing.
        """
        self.d = d
        self.arima_order = arima_order
        self.model = None
        self.results = None
        self.original_series = None
        self.diff_series = None
        self._weights = None
        self.kwargs = kwargs
    
    def _fractional_diff_weights(self, d: float, size: int) -> np.ndarray:
        """
        Compute weights for fractional differencing using binomial series.
        
        The weights follow: w_k = w_{k-1} * (k - 1 - d) / k
        
        Args:
            d: Fractional differencing parameter.
            size: Number of weights to compute.
            
        Returns:
            Array of weights.
        """
        weights = [1.0]
        for k in range(1, size):
            w = weights[-1] * (k - 1 - d) / k
            weights.append(w)
        return np.array(weights)
    
    def _apply_fractional_diff(self, series: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional differencing to a time series.
        
        Args:
            series: Input time series.
            d: Fractional differencing parameter.
            
        Returns:
            Fractionally differenced series.
        """
        n = len(series)
        weights = self._fractional_diff_weights(d, n)
        self._weights = weights
        
        diff_series = np.zeros(n)
        for t in range(n):
            # Apply convolution with weights
            diff_series[t] = np.sum(weights[:t+1][::-1] * series[:t+1])
        
        return diff_series
    
    def _apply_fractional_integration(self, diff_series: np.ndarray, d: float, 
                                       history_length: int) -> np.ndarray:
        """
        Apply fractional integration (inverse of differencing).
        
        This reconstructs the original scale from differenced predictions.
        
        Args:
            diff_series: Differenced series predictions.
            d: Fractional differencing parameter.
            history_length: Length of original series used for context.
            
        Returns:
            Integrated (original scale) predictions.
        """
        # For fractional integration, we use d -> -d
        n = len(diff_series)
        weights = self._fractional_diff_weights(-d, history_length + n)
        
        # We need the last value of the original series as starting point
        integrated = np.zeros(n)
        
        for t in range(n):
            # Cumulative integration using the original series context
            if t == 0:
                integrated[t] = diff_series[t]
            else:
                integrated[t] = diff_series[t] + np.sum(weights[1:t+1][::-1] * integrated[:t])
        
        return integrated
    
    def fit(self, df: pd.DataFrame) -> "FARIMAWrapper":
        """
        Fit FARIMA model.
        
        Args:
            df: DataFrame with columns 'ds' (datetime) and 'y' (value).
            
        Returns:
            Self for chaining.
        """
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        
        # Prepare time series
        ts = df.set_index("ds")["y"].values.astype(float)
        ts = np.nan_to_num(ts, nan=0.0)
        
        self.original_series = ts.copy()
        
        # Apply fractional differencing
        self.diff_series = self._apply_fractional_diff(ts, self.d)
        
        # Fit ARIMA on differenced series
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model = ARIMA(
                    self.diff_series,
                    order=self.arima_order
                )
                self.results = self.model.fit()
                
        except Exception as e:
            logger.warning(f"FARIMA ARIMA fitting failed: {e}. Using simpler order.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = ARIMA(
                    self.diff_series,
                    order=(1, 0, 1)
                )
                self.results = self.model.fit()
        
        return self
    
    def predict(self, periods: int = 1) -> float:
        """
        Predict the next value(s) using proper fractional integration.
        
        The inversion of fractional differencing (1-L)^d is done by solving:
        y_diff[t] = sum_{k=0}^{inf} w_k * y[t-k]  (where w_0 = 1)
        
        Which gives: y[t] = y_diff[t] - sum_{k=1}^{inf} w_k * y[t-k]
        
        Args:
            periods: Number of periods to predict (default: 1).
            
        Returns:
            Predicted value for the next period (in original scale).
        """
        # Forecast on differenced scale
        diff_forecast = self.results.forecast(steps=periods)
        
        predicted_diff = float(diff_forecast.iloc[-1] if hasattr(diff_forecast, 'iloc') 
                               else diff_forecast[-1])
        
        # Proper inversion of fractional differencing
        # y[t] = y_diff[t] - sum_{k=1}^{n} w_k * y[t-k]
        # where w_k are the fractional differencing weights
        
        # Use the stored weights from fitting
        if self._weights is None or len(self._weights) < 2:
            # Fallback to simple approach if weights not available
            return float(max(0, self.original_series[-1]))
        
        # Compute the sum of weighted past values
        n_history = min(len(self._weights) - 1, len(self.original_series))
        weighted_sum = 0.0
        for k in range(1, n_history + 1):
            weighted_sum += self._weights[k] * self.original_series[-k]
        
        # Invert: y[t] = y_diff[t] - sum(w_k * y[t-k])
        # Note: The differencing formula is y_diff[t] = sum(w_k * y[t-k])
        # So to invert: y[t] = y_diff[t] - (sum_{k>0} w_k * y[t-k])
        # This simplifies to: y[t] = y_diff[t] - weighted_sum
        predicted_value = predicted_diff - weighted_sum
        
        return float(max(0, predicted_value))  # Ensure non-negative


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_type: ModelType, **kwargs) -> Union[ProphetWrapper, FARIMAWrapper]:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ("prophet" or "farima").
        **kwargs: Additional model parameters.
        
    Returns:
        Initialized model wrapper.
    """
    if model_type == "prophet":
        return ProphetWrapper(**kwargs)
    elif model_type == "farima":
        return FARIMAWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'prophet' or 'farima'.")


# =============================================================================
# RECURSIVE FORECASTING (SINGLE MODEL)
# =============================================================================

def train_predict_recursive(
    df: pd.DataFrame,
    target_column: str,
    model_type: ModelType = "prophet",
    horizon: int = 4,
    mlflow_experiment_name: str = "github_activity_predictor",
    repo_name: str = "unknown",
    **model_kwargs
) -> Dict:
    """
    Train and predict using recursive multi-step forecasting.
    
    CRITICAL: This implementation has NO data leakage. At each step t,
    the model is trained ONLY on data from the start up to t (inclusive).
    Future data points are NEVER included in the training set.
    
    Args:
        df: DataFrame with columns 'ds' (datetime) and target_column.
        target_column: Name of the column to predict.
        model_type: Type of model to use ("prophet" or "farima").
        horizon: Number of weeks to forecast (test period).
        mlflow_experiment_name: Name of the MLflow experiment.
        repo_name: Name of the repository (for logging).
        **model_kwargs: Additional parameters passed to the model.
        
    Returns:
        Dictionary with results including predictions, metrics, and plot.
    """
    logger.info(
        f"Starting recursive forecasting: model={model_type}, "
        f"target={target_column}, horizon={horizon} weeks"
    )
    
    # Prepare data
    data = df[["ds", target_column]].copy()
    data = data.rename(columns={target_column: "y"})
    data = data.sort_values("ds").reset_index(drop=True)
    
    # Validate we have enough data
    min_train_size = 52  # Need at least 1 year for meaningful forecasting
    if len(data) < min_train_size + horizon:
        raise ValueError(
            f"Not enough data: {len(data)} weeks. "
            f"Need at least {min_train_size + horizon} weeks."
        )
    
    # Split: Train = all except last 'horizon' weeks, Test = last 'horizon' weeks
    train_size = len(data) - horizon
    
    # CRITICAL: train_df is the INITIAL training set (before any predictions)
    # This is the data that exists BEFORE the test period
    initial_train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()
    
    logger.info(f"Initial train size: {len(initial_train_df)} weeks, Test size: {len(test_df)} weeks")
    
    # Set up MLflow tracking (with quick timeout check)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_enabled = False
    
    try:
        import socket
        from urllib.parse import urlparse
        
        parsed = urlparse(mlflow_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow_enabled = True
            logger.info(f"MLflow tracking enabled at {mlflow_uri}")
        else:
            logger.warning(f"MLflow server not reachable at {mlflow_uri}. Continuing without tracking.")
    except Exception as e:
        logger.warning(f"MLflow connection check failed: {e}. Continuing without tracking.")
    
    # Storage for predictions
    predictions = []
    
    # Working copy of training data (will be extended with PREDICTIONS ONLY)
    # This NEVER includes actual future values - only our predictions
    working_train = initial_train_df.copy()
    
    # Recursive prediction loop
    run_context = mlflow.start_run() if mlflow_enabled else None
    
    try:
        if run_context:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("repo_name", repo_name)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("test_size", len(test_df))
            
            for key, value in model_kwargs.items():
                mlflow.log_param(f"model_{key}", value)
        
        for step in range(horizon):
            logger.info(f"Recursive step {step + 1}/{horizon}")
            
            # CRITICAL: Train model ONLY on working_train
            # working_train contains: initial_train + previous predictions (NOT actual future values)
            model = create_model(model_type, **model_kwargs)
            model.fit(working_train)
            
            # Predict only the next step
            predicted_value = model.predict(periods=1)
            
            # Ensure non-negative (counts can't be negative)
            predicted_value = max(0, predicted_value)
            
            # Get the actual date and value for this prediction (for evaluation only)
            next_date = test_df.iloc[step]["ds"]
            actual_value = test_df.iloc[step]["y"]
            
            predictions.append({
                "ds": next_date,
                "y_pred": predicted_value,
                "y_actual": actual_value,
                "step": step + 1
            })
            
            # CRITICAL: Add PREDICTION (not actual) to working training set
            # This is the "blind forecasting" approach - we don't know the actual value
            new_row = pd.DataFrame({
                "ds": [next_date],
                "y": [predicted_value]  # Use PREDICTION as pseudo ground truth
            })
            working_train = pd.concat([working_train, new_row], ignore_index=True)
            
            logger.debug(
                f"Step {step + 1}: predicted={predicted_value:.2f}, "
                f"actual={actual_value:.2f}"
            )
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Calculate metrics using SMAPE (not MAPE)
        y_true = pred_df["y_actual"].values
        y_pred = pred_df["y_pred"].values
        
        metrics = calculate_metrics(y_true, y_pred)
        
        logger.info(
            f"Metrics: RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, SMAPE={metrics['smape']:.2f}%"
        )
        
        # Log metrics to MLflow
        if run_context:
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("smape", metrics["smape"])
        
        # Create visualization
        fig = create_forecast_plot(
            train_df=initial_train_df,
            test_df=test_df,
            pred_df=pred_df,
            target_column=target_column,
            repo_name=repo_name,
            model_type=model_type,
            metrics=metrics
        )
        
        # Log artifact to MLflow
        if run_context:
            plot_path = f"/tmp/forecast_{repo_name}_{target_column}_{model_type}.html"
            fig.write_html(plot_path)
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)
            run_id = mlflow.active_run().info.run_id
        else:
            run_id = None
        
        results = {
            "predictions": pred_df,
            "metrics": metrics,
            "figure": fig,
            "train_df": initial_train_df,
            "test_df": test_df,
            "model_type": model_type,
            "target_column": target_column,
            "repo_name": repo_name,
            "horizon": horizon,
            "mlflow_run_id": run_id
        }
        
        return results
        
    finally:
        if run_context:
            mlflow.end_run()


# =============================================================================
# COMPARATIVE FORECASTING (PROPHET vs FARIMA)
# =============================================================================

def train_predict_comparative(
    df: pd.DataFrame,
    target_column: str,
    horizon: int = 4,
    repo_name: str = "unknown",
    prophet_kwargs: Optional[Dict] = None,
    farima_kwargs: Optional[Dict] = None
) -> Dict:
    """
    Train and predict using both Prophet and FARIMA for comparison.
    
    This function runs both models on the same data and returns comparative results.
    
    Args:
        df: DataFrame with columns 'ds' (datetime) and target_column.
        target_column: Name of the column to predict.
        horizon: Number of weeks to forecast.
        repo_name: Repository name for logging.
        prophet_kwargs: Additional kwargs for Prophet model.
        farima_kwargs: Additional kwargs for FARIMA model.
        
    Returns:
        Dictionary with:
        - prophet_predictions: DataFrame of Prophet predictions
        - farima_predictions: DataFrame of FARIMA predictions
        - prophet_metrics: Dict of Prophet metrics
        - farima_metrics: Dict of FARIMA metrics
        - comparison_figure: Plotly figure with both models
        - train_df, test_df: Original data splits
    """
    logger.info(f"Starting comparative forecasting: Prophet vs FARIMA, horizon={horizon}")
    
    prophet_kwargs = prophet_kwargs or {}
    farima_kwargs = farima_kwargs or {}
    
    # Prepare data
    data = df[["ds", target_column]].copy()
    data = data.rename(columns={target_column: "y"})
    data = data.sort_values("ds").reset_index(drop=True)
    
    # Validate data
    min_train_size = 52
    if len(data) < min_train_size + horizon:
        raise ValueError(
            f"Not enough data: {len(data)} weeks. "
            f"Need at least {min_train_size + horizon} weeks."
        )
    
    # Split data
    train_size = len(data) - horizon
    initial_train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()
    
    logger.info(f"Train: {len(initial_train_df)} weeks, Test: {len(test_df)} weeks")
    
    # Run Prophet
    logger.info("Training Prophet model...")
    prophet_predictions = _run_recursive_forecast(
        initial_train_df, test_df, "prophet", prophet_kwargs
    )
    
    # Run FARIMA
    logger.info("Training FARIMA model...")
    farima_predictions = _run_recursive_forecast(
        initial_train_df, test_df, "farima", farima_kwargs
    )
    
    # Calculate metrics for both
    y_true = test_df["y"].values
    
    prophet_metrics = calculate_metrics(y_true, prophet_predictions["y_pred"].values)
    farima_metrics = calculate_metrics(y_true, farima_predictions["y_pred"].values)
    
    logger.info(f"Prophet - RMSE: {prophet_metrics['rmse']:.4f}, SMAPE: {prophet_metrics['smape']:.2f}%")
    logger.info(f"FARIMA  - RMSE: {farima_metrics['rmse']:.4f}, SMAPE: {farima_metrics['smape']:.2f}%")
    
    # Create comparative visualization
    fig = create_comparative_plot(
        train_df=initial_train_df,
        test_df=test_df,
        prophet_pred_df=prophet_predictions,
        farima_pred_df=farima_predictions,
        target_column=target_column,
        repo_name=repo_name,
        prophet_metrics=prophet_metrics,
        farima_metrics=farima_metrics
    )
    
    return {
        "prophet_predictions": prophet_predictions,
        "farima_predictions": farima_predictions,
        "prophet_metrics": prophet_metrics,
        "farima_metrics": farima_metrics,
        "comparison_figure": fig,
        "train_df": initial_train_df,
        "test_df": test_df,
        "target_column": target_column,
        "repo_name": repo_name,
        "horizon": horizon
    }


def _run_recursive_forecast(
    initial_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: ModelType,
    model_kwargs: Dict
) -> pd.DataFrame:
    """
    Internal function to run recursive forecast for a single model.
    
    Args:
        initial_train_df: Initial training data.
        test_df: Test data for evaluation.
        model_type: Type of model.
        model_kwargs: Model parameters.
        
    Returns:
        DataFrame with predictions.
    """
    predictions = []
    working_train = initial_train_df.copy()
    horizon = len(test_df)
    
    for step in range(horizon):
        # Train model on current working data (NO future data)
        model = create_model(model_type, **model_kwargs)
        model.fit(working_train)
        
        # Predict next step
        predicted_value = max(0, model.predict(periods=1))
        
        # Record prediction
        next_date = test_df.iloc[step]["ds"]
        actual_value = test_df.iloc[step]["y"]
        
        predictions.append({
            "ds": next_date,
            "y_pred": predicted_value,
            "y_actual": actual_value,
            "step": step + 1
        })
        
        # Extend working train with prediction (NOT actual value)
        new_row = pd.DataFrame({"ds": [next_date], "y": [predicted_value]})
        working_train = pd.concat([working_train, new_row], ignore_index=True)
    
    return pd.DataFrame(predictions)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_forecast_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_column: str,
    repo_name: str,
    model_type: str,
    metrics: Dict
) -> go.Figure:
    """
    Create an interactive Plotly visualization of the forecast.
    """
    fig = go.Figure()
    
    # Training data (historical)
    fig.add_trace(go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="lines",
        name="Historical Data",
        line=dict(color="#2E86AB", width=2)
    ))
    
    # Actual test data
    fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="lines+markers",
        name="Actual (Test Period)",
        line=dict(color="#28A745", width=2),
        marker=dict(size=8)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=pred_df["ds"],
        y=pred_df["y_pred"],
        mode="lines+markers",
        name=f"{model_type.upper()} Prediction",
        line=dict(color="#DC3545", width=2, dash="dash"),
        marker=dict(size=8, symbol="x")
    ))
    
    # Vertical line at train/test split
    split_date = train_df["ds"].iloc[-1]
    split_date_str = split_date.isoformat() if hasattr(split_date, 'isoformat') else str(split_date)
    
    fig.add_shape(
        type="line",
        x0=split_date_str, x1=split_date_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="gray", dash="dot", width=1)
    )
    fig.add_annotation(
        x=split_date_str, y=1, yref="paper",
        text="Train/Test Split", showarrow=False, yshift=10
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{repo_name}</b> - {target_column} Forecast ({model_type.upper()})<br>"
                f"<sup>RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | "
                f"SMAPE: {metrics['smape']:.1f}%</sup>"
            ),
            x=0.5, xanchor="center"
        ),
        xaxis_title="Date",
        yaxis_title=target_column.replace("_", " ").title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


def create_comparative_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prophet_pred_df: pd.DataFrame,
    farima_pred_df: pd.DataFrame,
    target_column: str,
    repo_name: str,
    prophet_metrics: Dict,
    farima_metrics: Dict
) -> go.Figure:
    """
    Create a comparative visualization with Prophet and FARIMA predictions.
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="lines",
        name="Historical Data",
        line=dict(color="#2E86AB", width=2)
    ))
    
    # Actual test data (Ground Truth)
    fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="lines+markers",
        name="Ground Truth",
        line=dict(color="#28A745", width=3),
        marker=dict(size=10, symbol="circle")
    ))
    
    # Prophet predictions
    fig.add_trace(go.Scatter(
        x=prophet_pred_df["ds"],
        y=prophet_pred_df["y_pred"],
        mode="lines+markers",
        name=f"Prophet (SMAPE: {prophet_metrics['smape']:.1f}%)",
        line=dict(color="#FF6B35", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond")
    ))
    
    # FARIMA predictions
    fig.add_trace(go.Scatter(
        x=farima_pred_df["ds"],
        y=farima_pred_df["y_pred"],
        mode="lines+markers",
        name=f"FARIMA (SMAPE: {farima_metrics['smape']:.1f}%)",
        line=dict(color="#9B59B6", width=2, dash="dot"),
        marker=dict(size=8, symbol="x")
    ))
    
    # Vertical line at split
    split_date = train_df["ds"].iloc[-1]
    split_date_str = split_date.isoformat() if hasattr(split_date, 'isoformat') else str(split_date)
    
    fig.add_shape(
        type="line",
        x0=split_date_str, x1=split_date_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="gray", dash="dot", width=1)
    )
    fig.add_annotation(
        x=split_date_str, y=1, yref="paper",
        text="Forecast Start", showarrow=False, yshift=10
    )
    
    # Layout with metrics comparison in subtitle
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{repo_name}</b> - {target_column}: Prophet vs FARIMA Comparison<br>"
                f"<sup>Prophet - RMSE: {prophet_metrics['rmse']:.2f}, SMAPE: {prophet_metrics['smape']:.1f}% | "
                f"FARIMA - RMSE: {farima_metrics['rmse']:.2f}, SMAPE: {farima_metrics['smape']:.1f}%</sup>"
            ),
            x=0.5, xanchor="center"
        ),
        xaxis_title="Date",
        yaxis_title=target_column.replace("_", " ").title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_full_analysis(
    df: pd.DataFrame,
    horizon: int = 4,
    repo_name: str = "unknown",
    target_columns: Optional[List[str]] = None,
    compare_models: bool = True
) -> Dict[str, Dict]:
    """
    Run forecasting analysis on multiple target columns.
    
    Args:
        df: Full DataFrame with multiple metrics.
        horizon: Forecast horizon in weeks.
        repo_name: Repository name.
        target_columns: List of columns to forecast.
        compare_models: If True, run both Prophet and FARIMA.
        
    Returns:
        Dictionary mapping column names to their results.
    """
    if target_columns is None:
        target_columns = ["commits", "new_stars", "issues_opened", "prs_opened"]
    
    available_columns = [col for col in target_columns if col in df.columns]
    
    results = {}
    for col in available_columns:
        logger.info(f"\n{'='*60}\nAnalyzing: {col}\n{'='*60}")
        try:
            if compare_models:
                result = train_predict_comparative(
                    df=df,
                    target_column=col,
                    horizon=horizon,
                    repo_name=repo_name
                )
            else:
                result = train_predict_recursive(
                    df=df,
                    target_column=col,
                    model_type="prophet",
                    horizon=horizon,
                    repo_name=repo_name
                )
            results[col] = result
        except Exception as e:
            logger.error(f"Failed to analyze {col}: {e}")
            results[col] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage / testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.etl import load_repository_data
    
    repo_path = "repositories/3b1b__manim"
    repo_name = Path(repo_path).name
    
    print(f"\n{'='*60}")
    print(f"Model Engine Test - Repository: {repo_name}")
    print(f"{'='*60}\n")
    
    try:
        df = load_repository_data(repo_path)
        print(f"Loaded data: {len(df)} weeks")
        
        # Run comparative forecast
        results = train_predict_comparative(
            df=df,
            target_column="commits",
            horizon=8,
            repo_name=repo_name
        )
        
        print("\n=== Prophet Predictions ===")
        print(results["prophet_predictions"])
        print(f"\nMetrics: {results['prophet_metrics']}")
        
        print("\n=== FARIMA Predictions ===")
        print(results["farima_predictions"])
        print(f"\nMetrics: {results['farima_metrics']}")
        
        # Show plot
        results["comparison_figure"].show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
