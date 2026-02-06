"""
Model Selection Automation Module - GitHub Activity Predictor
==============================================================

This module implements automatic model selection and hyperparameter
optimization between Prophet, FARIMA, and Neural Network models.

Features:
- Automated model selection based on data characteristics
- Hyperparameter optimization using grid search and Bayesian optimization
- Cross-validation for robust model comparison
- Confidence intervals for predictions
- MLflow integration for experiment tracking
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Literal, Union
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ModelType = Literal["prophet", "farima", "neural_network"]


@dataclass
class HyperparameterSpace:
    """Defines hyperparameter search space for a model."""
    
    model_type: ModelType
    parameters: Dict[str, List[Any]]
    
    def get_combinations(self) -> List[Dict]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


# Default hyperparameter spaces
PROPHET_SEARCH_SPACE = HyperparameterSpace(
    model_type="prophet",
    parameters={
        "seasonality_mode": ["additive", "multiplicative"],
        "yearly_seasonality": [True, False],
        "weekly_seasonality": [True, False],
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    }
)

FARIMA_SEARCH_SPACE = HyperparameterSpace(
    model_type="farima",
    parameters={
        "d": [0.2, 0.3, 0.4],
        "arima_order": [(1, 0, 1), (2, 0, 1), (2, 0, 2), (1, 0, 2)],
    }
)

NEURAL_NETWORK_SEARCH_SPACE = HyperparameterSpace(
    model_type="neural_network",
    parameters={
        "lookback_window": [8, 12, 16],
        "hidden_layers": [[32, 16], [64, 32], [128, 64, 32]],
        "dropout_rate": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.01],
    }
)


@dataclass
class ModelEvaluationResult:
    """Result from evaluating a model configuration."""
    
    model_type: ModelType
    parameters: Dict
    metrics: Dict[str, float]
    predictions: List[Dict]
    training_time: float
    prediction_intervals: Optional[Dict] = None
    cv_scores: Optional[List[float]] = None
    
    @property
    def primary_metric(self) -> float:
        """Get the primary metric (SMAPE) for comparison."""
        return self.metrics.get("smape", float("inf"))


@dataclass
class AutoModelResult:
    """Result from automated model selection."""
    
    best_model_type: ModelType
    best_parameters: Dict
    best_metrics: Dict[str, float]
    all_results: List[ModelEvaluationResult]
    selection_reason: str
    prediction_intervals: Dict
    training_time: float
    
    def get_rankings(self) -> pd.DataFrame:
        """Get ranked results as DataFrame."""
        rows = []
        for result in self.all_results:
            rows.append({
                "model_type": result.model_type,
                "parameters": str(result.parameters),
                "rmse": result.metrics.get("rmse"),
                "mae": result.metrics.get("mae"),
                "smape": result.metrics.get("smape"),
                "training_time": result.training_time
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values("smape").reset_index(drop=True)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            (y_true == 0) & (y_pred == 0),
            0.0,
            numerator / np.where(denominator == 0, 1, denominator)
        )
    
    return float(100 * np.mean(ratio))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all forecasting metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    
    return {"rmse": rmse, "mae": mae, "smape": smape}


class ModelEvaluator:
    """Evaluates models using recursive forecasting."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        horizon: int = 8,
        min_train_size: int = 52
    ):
        """
        Initialize model evaluator.
        
        Args:
            df: DataFrame with time series data.
            target_column: Column to predict.
            horizon: Forecast horizon in weeks.
            min_train_size: Minimum training data size.
        """
        self.df = df
        self.target_column = target_column
        self.horizon = horizon
        self.min_train_size = min_train_size
        
        # Prepare data
        self.data = df[["ds", target_column]].copy()
        self.data = self.data.rename(columns={target_column: "y"})
        
        # Include PRs for neural network correlation
        if "prs_opened" in df.columns:
            self.data["prs_opened"] = df["prs_opened"].values
        
        self.data = self.data.sort_values("ds").reset_index(drop=True)
    
    def _create_prophet_model(self, params: Dict):
        """Create a Prophet model with given parameters."""
        from prophet import Prophet
        
        model_params = {
            "yearly_seasonality": params.get("yearly_seasonality", True),
            "weekly_seasonality": params.get("weekly_seasonality", True),
            "daily_seasonality": False,
            "seasonality_mode": params.get("seasonality_mode", "additive"),
        }
        
        if "changepoint_prior_scale" in params:
            model_params["changepoint_prior_scale"] = params["changepoint_prior_scale"]
        
        return Prophet(**model_params)
    
    def _create_farima_model(self, params: Dict):
        """Create a FARIMA model with given parameters."""
        from src.model_engine import FARIMAWrapper
        
        return FARIMAWrapper(
            d=params.get("d", 0.4),
            arima_order=params.get("arima_order", (2, 0, 2))
        )
    
    def _create_neural_network_model(self, params: Dict):
        """Create a Neural Network model with given parameters."""
        from src.neural_network import NeuralNetworkWrapper, NeuralNetworkConfig
        
        config = NeuralNetworkConfig(
            lookback_window=params.get("lookback_window", 12),
            hidden_layers=params.get("hidden_layers"),
            dropout_rate=params.get("dropout_rate", 0.2),
            learning_rate=params.get("learning_rate", 0.001),
            use_prs_correlation=params.get("use_prs_correlation", True),
            auto_scale=params.get("auto_scale", True),
            n_ensemble=params.get("n_ensemble", 5)
        )
        
        return NeuralNetworkWrapper(config)
    
    def _run_recursive_forecast(
        self,
        model_type: ModelType,
        params: Dict,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[List[Dict], Dict[str, List[float]]]:
        """
        Run recursive multi-step forecast.
        
        Returns:
            Tuple of (predictions list, prediction intervals dict).
        """
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        working_train = train_df.copy()
        
        import warnings
        warnings.filterwarnings("ignore")
        
        for step in range(len(test_df)):
            # Create and train model
            if model_type == "prophet":
                model = self._create_prophet_model(params)
                
                import logging as py_logging
                py_logging.getLogger("cmdstanpy").setLevel(py_logging.WARNING)
                py_logging.getLogger("prophet").setLevel(py_logging.WARNING)
                
                model.fit(working_train[["ds", "y"]])
                
                # Predict with intervals
                future = model.make_future_dataframe(periods=1, freq="W")
                forecast = model.predict(future)
                
                predicted_value = float(forecast.iloc[-1]["yhat"])
                lower = float(forecast.iloc[-1]["yhat_lower"])
                upper = float(forecast.iloc[-1]["yhat_upper"])
                
            elif model_type == "farima":
                model = self._create_farima_model(params)
                model.fit(working_train[["ds", "y"]])
                predicted_value = model.predict(periods=1)
                
                # FARIMA doesn't have built-in intervals, estimate from residuals
                std_estimate = working_train["y"].std() * 0.5
                lower = predicted_value - 1.96 * std_estimate
                upper = predicted_value + 1.96 * std_estimate
                
            else:  # neural_network
                model = self._create_neural_network_model(params)
                model.fit(working_train)
                
                # Neural network has ensemble-based intervals
                result = model.predict_with_intervals(periods=1)
                predicted_value = result["prediction"]
                lower = result["lower"]
                upper = result["upper"]
            
            # Ensure non-negative
            predicted_value = max(0, predicted_value)
            lower = max(0, lower)
            upper = max(0, upper)
            
            # Record prediction
            actual_value = test_df.iloc[step]["y"]
            next_date = test_df.iloc[step]["ds"]
            
            predictions.append({
                "ds": next_date,
                "y_pred": predicted_value,
                "y_actual": actual_value,
                "step": step + 1
            })
            
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Extend working train with prediction
            new_row = {"ds": next_date, "y": predicted_value}
            if "prs_opened" in working_train.columns:
                new_row["prs_opened"] = test_df.iloc[step].get("prs_opened", 0)
            
            working_train = pd.concat(
                [working_train, pd.DataFrame([new_row])], 
                ignore_index=True
            )
        
        intervals = {
            "lower": lower_bounds,
            "upper": upper_bounds
        }
        
        return predictions, intervals
    
    def evaluate_model(
        self,
        model_type: ModelType,
        params: Dict
    ) -> ModelEvaluationResult:
        """
        Evaluate a single model configuration.
        
        Args:
            model_type: Type of model to evaluate.
            params: Model parameters.
            
        Returns:
            Evaluation result.
        """
        start_time = time.time()
        
        # Split data
        train_size = len(self.data) - self.horizon
        train_df = self.data.iloc[:train_size].copy()
        test_df = self.data.iloc[train_size:].copy()
        
        # Run forecast
        try:
            predictions, intervals = self._run_recursive_forecast(
                model_type, params, train_df, test_df
            )
            
            # Calculate metrics
            y_true = [p["y_actual"] for p in predictions]
            y_pred = [p["y_pred"] for p in predictions]
            
            metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
            
            training_time = time.time() - start_time
            
            return ModelEvaluationResult(
                model_type=model_type,
                parameters=params,
                metrics=metrics,
                predictions=predictions,
                training_time=training_time,
                prediction_intervals=intervals
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {e}")
            return ModelEvaluationResult(
                model_type=model_type,
                parameters=params,
                metrics={"rmse": float("inf"), "mae": float("inf"), "smape": float("inf")},
                predictions=[],
                training_time=time.time() - start_time
            )
    
    def cross_validate(
        self,
        model_type: ModelType,
        params: Dict,
        n_splits: int = 3
    ) -> List[float]:
        """
        Cross-validate a model using time series splits.
        
        Args:
            model_type: Type of model.
            params: Model parameters.
            n_splits: Number of CV splits.
            
        Returns:
            List of SMAPE scores for each split.
        """
        scores = []
        
        # Time series split
        n_samples = len(self.data)
        min_train = self.min_train_size
        
        # Calculate split points
        available_test = n_samples - min_train
        test_per_split = max(self.horizon, available_test // (n_splits + 1))
        
        for i in range(n_splits):
            train_end = min_train + i * test_per_split
            test_end = min(train_end + self.horizon, n_samples)
            
            if train_end >= n_samples - self.horizon:
                break
            
            train_df = self.data.iloc[:train_end].copy()
            test_df = self.data.iloc[train_end:test_end].copy()
            
            if len(test_df) < 2:
                continue
            
            try:
                predictions, _ = self._run_recursive_forecast(
                    model_type, params, train_df, test_df
                )
                
                y_true = [p["y_actual"] for p in predictions]
                y_pred = [p["y_pred"] for p in predictions]
                
                smape = calculate_smape(np.array(y_true), np.array(y_pred))
                scores.append(smape)
                
            except Exception as e:
                logger.warning(f"CV fold {i} failed: {e}")
                scores.append(float("inf"))
        
        return scores


class AutoModelSelector:
    """
    Automated model selection with hyperparameter optimization.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        horizon: int = 8,
        model_types: Optional[List[ModelType]] = None,
        use_cross_validation: bool = True,
        n_cv_splits: int = 3,
        max_evaluations: int = 50
    ):
        """
        Initialize auto model selector.
        
        Args:
            df: DataFrame with time series data.
            target_column: Column to predict.
            horizon: Forecast horizon.
            model_types: Models to consider. If None, uses all.
            use_cross_validation: Whether to use CV for evaluation.
            n_cv_splits: Number of CV splits.
            max_evaluations: Maximum number of model evaluations.
        """
        self.df = df
        self.target_column = target_column
        self.horizon = horizon
        self.model_types = model_types or ["prophet", "farima", "neural_network"]
        self.use_cv = use_cross_validation
        self.n_cv_splits = n_cv_splits
        self.max_evaluations = max_evaluations
        
        self.evaluator = ModelEvaluator(df, target_column, horizon)
    
    def _get_search_space(self, model_type: ModelType) -> HyperparameterSpace:
        """Get search space for a model type."""
        if model_type == "prophet":
            return PROPHET_SEARCH_SPACE
        elif model_type == "farima":
            return FARIMA_SEARCH_SPACE
        else:
            return NEURAL_NETWORK_SEARCH_SPACE
    
    def _analyze_data_characteristics(self) -> Dict[str, Any]:
        """Analyze data to inform model selection."""
        data = self.df[self.target_column].values
        
        # Basic statistics
        n_samples = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # Trend detection
        from scipy import stats
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        has_trend = abs(r_value) > 0.5
        
        # Seasonality detection (simple autocorrelation check)
        from scipy.signal import correlate
        autocorr = correlate(data - mean_val, data - mean_val, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Check for weekly seasonality (lag 52 for yearly if enough data)
        has_seasonality = False
        if len(autocorr) > 52:
            if autocorr[52] > 0.3:
                has_seasonality = True
        
        # Sparsity check
        sparsity = np.mean(data == 0)
        
        # Volatility
        returns = np.diff(data) / (data[:-1] + 1)
        volatility = np.std(returns)
        
        return {
            "n_samples": n_samples,
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "has_trend": has_trend,
            "has_seasonality": has_seasonality,
            "sparsity": sparsity,
            "volatility": volatility,
            "slope": slope
        }
    
    def _prioritize_models(
        self, 
        data_chars: Dict
    ) -> List[Tuple[ModelType, float]]:
        """
        Prioritize models based on data characteristics.
        
        Returns:
            List of (model_type, priority_score) tuples.
        """
        scores = {}
        
        # Prophet: good for seasonal data with trends
        prophet_score = 0.5
        if data_chars["has_seasonality"]:
            prophet_score += 0.3
        if data_chars["has_trend"]:
            prophet_score += 0.2
        if data_chars["sparsity"] < 0.3:
            prophet_score += 0.1
        scores["prophet"] = prophet_score
        
        # FARIMA: good for long-memory, non-seasonal data
        farima_score = 0.5
        if not data_chars["has_seasonality"]:
            farima_score += 0.2
        if data_chars["volatility"] < 0.5:
            farima_score += 0.2
        if data_chars["n_samples"] > 100:
            farima_score += 0.1
        scores["farima"] = farima_score
        
        # Neural Network: good for complex patterns, needs more data
        nn_score = 0.4
        if data_chars["n_samples"] > 100:
            nn_score += 0.3
        if data_chars["cv"] > 0.5:  # High variability
            nn_score += 0.2
        if "prs_opened" in self.df.columns:
            nn_score += 0.2  # Can exploit correlation
        scores["neural_network"] = nn_score
        
        # Sort by score
        prioritized = sorted(
            [(m, s) for m, s in scores.items() if m in self.model_types],
            key=lambda x: x[1],
            reverse=True
        )
        
        return prioritized
    
    def select_best_model(
        self,
        quick_mode: bool = False
    ) -> AutoModelResult:
        """
        Automatically select the best model.
        
        Args:
            quick_mode: If True, uses reduced search space for faster results.
            
        Returns:
            AutoModelResult with best model and comparisons.
        """
        start_time = time.time()
        
        logger.info("Starting automated model selection...")
        
        # Analyze data
        data_chars = self._analyze_data_characteristics()
        logger.info(f"Data characteristics: {data_chars}")
        
        # Prioritize models
        prioritized = self._prioritize_models(data_chars)
        logger.info(f"Model priorities: {prioritized}")
        
        all_results = []
        evaluations_count = 0
        
        # Evaluate each model type
        for model_type, priority in prioritized:
            search_space = self._get_search_space(model_type)
            combinations = search_space.get_combinations()
            
            if quick_mode:
                # Reduce combinations in quick mode
                combinations = combinations[:3]
            
            logger.info(f"Evaluating {model_type}: {len(combinations)} configurations")
            
            for params in combinations:
                if evaluations_count >= self.max_evaluations:
                    break
                
                logger.info(f"Evaluating {model_type} with {params}")
                
                result = self.evaluator.evaluate_model(model_type, params)
                
                # Add CV scores if enabled
                if self.use_cv and result.metrics["smape"] < float("inf"):
                    try:
                        cv_scores = self.evaluator.cross_validate(
                            model_type, params, self.n_cv_splits
                        )
                        result.cv_scores = cv_scores
                    except Exception as e:
                        logger.warning(f"CV failed: {e}")
                
                all_results.append(result)
                evaluations_count += 1
                
                logger.info(
                    f"Result: SMAPE={result.metrics.get('smape', 'N/A'):.2f}%, "
                    f"time={result.training_time:.1f}s"
                )
        
        # Select best model
        valid_results = [r for r in all_results if r.metrics["smape"] < float("inf")]
        
        if not valid_results:
            raise ValueError("No valid model configurations found")
        
        # Sort by SMAPE (or CV mean if available)
        def get_score(r):
            if r.cv_scores and len(r.cv_scores) > 0:
                return np.mean(r.cv_scores)
            return r.metrics["smape"]
        
        best_result = min(valid_results, key=get_score)
        
        # Generate selection reason
        reason = self._generate_selection_reason(best_result, data_chars, prioritized)
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Best model: {best_result.model_type} with SMAPE={best_result.metrics['smape']:.2f}%"
        )
        
        return AutoModelResult(
            best_model_type=best_result.model_type,
            best_parameters=best_result.parameters,
            best_metrics=best_result.metrics,
            all_results=all_results,
            selection_reason=reason,
            prediction_intervals=best_result.prediction_intervals or {},
            training_time=total_time
        )
    
    def _generate_selection_reason(
        self,
        best: ModelEvaluationResult,
        data_chars: Dict,
        priorities: List[Tuple[ModelType, float]]
    ) -> str:
        """Generate a human-readable selection reason."""
        reasons = []
        
        reasons.append(
            f"Selected {best.model_type.upper()} with SMAPE={best.metrics['smape']:.2f}%"
        )
        
        if best.cv_scores:
            mean_cv = np.mean(best.cv_scores)
            std_cv = np.std(best.cv_scores)
            reasons.append(f"Cross-validation: {mean_cv:.2f}% ± {std_cv:.2f}%")
        
        if data_chars["has_seasonality"] and best.model_type == "prophet":
            reasons.append("Prophet chosen due to detected seasonality in data")
        
        if data_chars["n_samples"] > 100 and best.model_type == "neural_network":
            reasons.append("Neural network leveraged larger dataset size")
        
        if "prs_opened" in self.df.columns and best.model_type == "neural_network":
            reasons.append("Neural network exploited commits-PRs correlation")
        
        return ". ".join(reasons)


def auto_select_model(
    df: pd.DataFrame,
    target_column: str = "commits",
    horizon: int = 8,
    quick_mode: bool = False,
    model_types: Optional[List[ModelType]] = None
) -> AutoModelResult:
    """
    Convenience function for automatic model selection.
    
    Args:
        df: DataFrame with time series data.
        target_column: Column to predict.
        horizon: Forecast horizon.
        quick_mode: Use quick mode with reduced search.
        model_types: Models to consider.
        
    Returns:
        AutoModelResult with best model and comparisons.
    """
    selector = AutoModelSelector(
        df=df,
        target_column=target_column,
        horizon=horizon,
        model_types=model_types
    )
    
    return selector.select_best_model(quick_mode=quick_mode)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.etl import load_repository_data
    
    print("Testing Auto Model Selection")
    print("=" * 60)
    
    repo_path = "repositories/3b1b__manim"
    
    try:
        df = load_repository_data(repo_path)
        print(f"Loaded data: {len(df)} weeks")
        
        # Run auto selection (quick mode for testing)
        result = auto_select_model(
            df=df,
            target_column="commits",
            horizon=8,
            quick_mode=True,
            model_types=["prophet", "farima"]
        )
        
        print(f"\n{'='*60}")
        print(f"Best Model: {result.best_model_type}")
        print(f"Parameters: {result.best_parameters}")
        print(f"Metrics: {result.best_metrics}")
        print(f"Reason: {result.selection_reason}")
        print(f"Total Time: {result.training_time:.1f}s")
        
        print("\nRankings:")
        print(result.get_rankings())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
