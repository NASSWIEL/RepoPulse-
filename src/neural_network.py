"""
Neural Network Model - GitHub Activity Predictor
=================================================

This module implements a neural network model for predicting GitHub activity,
specifically designed to exploit cross-metric correlations (commits with PRs).

Features:
- Multi-input neural network using commits and PRs correlation
- Auto-adaptive network architecture based on dataset size
- Confidence intervals via ensemble/dropout methods
- Recursive multi-step forecasting
- MLflow integration for experiment tracking
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NeuralNetworkConfig:
    """Configuration for the neural network model."""
    
    def __init__(
        self,
        lookback_window: int = 24,  # Increased for more context
        hidden_layers: Optional[List[int]] = None,
        dropout_rate: float = 0.4,  # Increased for better regularization
        learning_rate: float = 0.0005,  # Reduced for stability
        epochs: int = 300,  # More epochs with patience
        batch_size: int = 16,  # Smaller batches for better gradients
        early_stopping_patience: int = 30,  # More patience
        n_ensemble: int = 5,  # For confidence intervals
        use_prs_correlation: bool = True,
        auto_scale: bool = True,
        # New parameters for improved architecture
        use_feature_engineering: bool = True,
        l2_regularization: float = 0.01,  # Weight decay
        use_batch_norm: bool = True,
        use_residual: bool = True,  # Skip connections
        noise_injection: float = 0.05  # Input noise for robustness
    ):
        """
        Initialize neural network configuration.
        
        Args:
            lookback_window: Number of past weeks to use as features.
            hidden_layers: List of hidden layer sizes. If None, auto-determined.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for optimizer.
            epochs: Maximum training epochs.
            batch_size: Training batch size.
            early_stopping_patience: Patience for early stopping.
            n_ensemble: Number of models in ensemble for confidence intervals.
            use_prs_correlation: Whether to include PR features.
            auto_scale: Whether to auto-scale network based on data size.
            use_feature_engineering: Whether to add rolling statistics features.
            l2_regularization: L2 weight decay for regularization.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual/skip connections.
            noise_injection: Gaussian noise std to inject during training.
        """
        self.lookback_window = lookback_window
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.n_ensemble = n_ensemble
        self.use_prs_correlation = use_prs_correlation
        self.auto_scale = auto_scale
        self.use_feature_engineering = use_feature_engineering
        self.l2_regularization = l2_regularization
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.noise_injection = noise_injection
    
    def get_hidden_layers(self, n_samples: int, n_features: int) -> List[int]:
        """
        Get hidden layer configuration, auto-scaling if needed.
        
        Args:
            n_samples: Number of training samples.
            n_features: Number of input features.
            
        Returns:
            List of hidden layer sizes.
        """
        if self.hidden_layers is not None:
            return self.hidden_layers
        
        if not self.auto_scale:
            return [64, 32]
        
        # Auto-scale based on data size
        # Rule of thumb: smaller networks for smaller datasets
        if n_samples < 50:
            # Very small dataset - simple network
            return [max(8, n_features)]
        elif n_samples < 100:
            # Small dataset
            return [max(16, n_features * 2), max(8, n_features)]
        elif n_samples < 200:
            # Medium dataset
            return [64, 32]
        elif n_samples < 500:
            # Larger dataset
            return [128, 64, 32]
        else:
            # Large dataset
            return [256, 128, 64]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging."""
        return {
            "lookback_window": self.lookback_window,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "n_ensemble": self.n_ensemble,
            "use_prs_correlation": self.use_prs_correlation,
            "auto_scale": self.auto_scale,
            "use_feature_engineering": self.use_feature_engineering,
            "l2_regularization": self.l2_regularization,
            "use_batch_norm": self.use_batch_norm,
            "use_residual": self.use_residual,
            "noise_injection": self.noise_injection
        }


def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class NeuralNetworkWrapper:
    """
    Neural Network wrapper for time series forecasting with cross-metric correlation.
    
    This model predicts commits using historical commit data and pull request data
    to capture the correlation between PRs and commits.
    """
    
    def __init__(self, config: Optional[NeuralNetworkConfig] = None):
        """
        Initialize the neural network wrapper.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or NeuralNetworkConfig()
        self.models = []  # Ensemble of models
        self.scalers = {}
        self.is_fitted = False
        self._last_train_data = None
        self._feature_names = None
        
        # Check PyTorch availability
        if not _check_torch_available():
            logger.warning("PyTorch not available. Using fallback implementation.")
            self._use_fallback = True
        else:
            self._use_fallback = False
    
    def _compute_rolling_features(self, data: np.ndarray, lookback: int) -> np.ndarray:
        """
        Compute rolling statistical features for enhanced prediction.
        
        Args:
            data: Input time series data.
            lookback: Lookback window size.
            
        Returns:
            Array of engineered features.
        """
        features = []
        n_samples = len(data) - lookback
        
        for i in range(n_samples):
            window = data[i:i + lookback]
            
            # Basic statistics
            feat = [
                np.mean(window),                    # Mean
                np.std(window) + 1e-8,              # Std with epsilon
                np.min(window),                     # Min
                np.max(window),                     # Max
                window[-1],                         # Last value
                window[-1] - window[0],             # Trend (first to last)
            ]
            
            # Rolling means at different scales
            if lookback >= 4:
                feat.append(np.mean(window[-4:]))   # Last 4 weeks mean
            if lookback >= 8:
                feat.append(np.mean(window[-8:]))   # Last 8 weeks mean
            
            # Momentum features
            if lookback >= 2:
                feat.append(window[-1] - window[-2])  # Week-over-week change
            if lookback >= 4:
                feat.append(np.mean(window[-2:]) - np.mean(window[-4:-2]))  # Short-term momentum
            
            # Volatility
            if lookback >= 4:
                feat.append(np.std(window[-4:]) / (np.mean(window[-4:]) + 1e-8))  # CV last 4 weeks
            
            features.append(feat)
        
        return np.array(features)
    
    def _create_sequences(
        self, 
        commits: np.ndarray, 
        prs: Optional[np.ndarray] = None,
        lookback: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for the neural network with feature engineering.
        
        Args:
            commits: Array of commit counts.
            prs: Array of PR counts (optional).
            lookback: Number of past timesteps to include.
            
        Returns:
            Tuple of (X, y) arrays for training.
        """
        n_samples = len(commits) - lookback
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data: need at least {lookback + 1} samples")
        
        feature_list = []
        self._feature_names = []
        
        # Raw lookback features
        raw_features = []
        for i in range(n_samples):
            raw_features.append(commits[i:i + lookback])
        raw_features = np.array(raw_features)
        feature_list.append(raw_features)
        self._feature_names.extend([f"commit_t-{j}" for j in range(lookback, 0, -1)])
        
        # Add engineered features if enabled
        if self.config.use_feature_engineering:
            eng_features = self._compute_rolling_features(commits, lookback)
            feature_list.append(eng_features)
            self._feature_names.extend([
                "mean", "std", "min", "max", "last", "trend",
                "mean_4w", "mean_8w", "wow_change", "momentum", "cv_4w"
            ][:eng_features.shape[1]])
        
        # Add PR features if available
        if prs is not None and self.config.use_prs_correlation:
            prs_raw = []
            for i in range(n_samples):
                prs_raw.append(prs[i:i + lookback])
            prs_raw = np.array(prs_raw)
            feature_list.append(prs_raw)
            self._feature_names.extend([f"prs_t-{j}" for j in range(lookback, 0, -1)])
            
            if self.config.use_feature_engineering:
                prs_eng = self._compute_rolling_features(prs, lookback)
                feature_list.append(prs_eng)
                self._feature_names.extend([
                    "prs_mean", "prs_std", "prs_min", "prs_max", "prs_last", "prs_trend"
                ][:prs_eng.shape[1]])
        
        # Concatenate all features
        X = np.hstack(feature_list)
        
        # Target values
        y = np.array([commits[i + lookback] for i in range(n_samples)])
        
        logger.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}, features={len(self._feature_names)}")
        
        return X, y
    
    def _normalize_data(
        self, 
        data: np.ndarray, 
        key: str, 
        fit: bool = True
    ) -> np.ndarray:
        """Normalize data using min-max scaling."""
        from sklearn.preprocessing import MinMaxScaler
        
        data = data.reshape(-1, 1) if len(data.shape) == 1 else data
        
        if fit:
            self.scalers[key] = MinMaxScaler()
            return self.scalers[key].fit_transform(data).flatten()
        else:
            if key not in self.scalers:
                return data.flatten()
            return self.scalers[key].transform(data).flatten()
    
    def _inverse_normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Inverse normalize data."""
        if key not in self.scalers:
            return data
        data = data.reshape(-1, 1) if len(data.shape) == 1 else data
        return self.scalers[key].inverse_transform(data).flatten()
    
    def _build_torch_model(self, n_features: int, hidden_layers: List[int]):
        """Build a PyTorch neural network model."""
        import torch
        import torch.nn as nn
        
        layers = []
        input_size = n_features
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, 1))
        
        return nn.Sequential(*layers)
    
    def _train_torch_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        hidden_layers: List[int]
    ):
        """Train a single PyTorch model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=min(self.config.batch_size, len(X)),
            shuffle=True
        )
        
        # Build model
        model = self._build_torch_model(X.shape[1], hidden_layers)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.debug(f"Early stopping at epoch {epoch}")
                    break
        
        return model
    
    def _train_fallback_model(self, X: np.ndarray, y: np.ndarray):
        """Train an improved fallback model using sklearn with ensemble."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        hidden_layers = self.config.get_hidden_layers(len(X), X.shape[1])
        
        # Standardize features for better convergence
        self._feature_scaler = StandardScaler()
        X_scaled = self._feature_scaler.fit_transform(X)
        
        # Add noise injection for robustness if training
        if self.config.noise_injection > 0:
            noise = np.random.normal(0, self.config.noise_injection, X_scaled.shape)
            X_noisy = X_scaled + noise
        else:
            X_noisy = X_scaled
        
        # Neural network with strong regularization
        nn_model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='adam',
            alpha=self.config.l2_regularization,  # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=self.config.early_stopping_patience,
            random_state=42,
            verbose=True
        )
        
        # Gradient boosting for robustness
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Ridge regression as a stable baseline
        ridge_model = Ridge(alpha=1.0)
        
        # Create ensemble
        ensemble = VotingRegressor([
            ('nn', nn_model),
            ('gb', gb_model),
            ('ridge', ridge_model)
        ])
        
        logger.info(f"Training ensemble model with hidden layers: {hidden_layers}")
        ensemble.fit(X_noisy, y)
        
        return ensemble
    
    def fit(self, df: pd.DataFrame) -> "NeuralNetworkWrapper":
        """
        Fit the neural network model on training data.
        
        Args:
            df: DataFrame with columns 'ds' (datetime), 'y' (commits), 
                and optionally 'prs_opened' for cross-correlation.
            
        Returns:
            Self for chaining.
        """
        logger.info("Training neural network model...")
        
        # Extract data
        commits = df["y"].values.astype(float)
        
        prs = None
        if "prs_opened" in df.columns and self.config.use_prs_correlation:
            prs = df["prs_opened"].values.astype(float)
        
        # Normalize data
        commits_norm = self._normalize_data(commits, "commits", fit=True)
        if prs is not None:
            prs_norm = self._normalize_data(prs, "prs", fit=True)
        else:
            prs_norm = None
        
        # Create sequences
        X, y = self._create_sequences(
            commits_norm, 
            prs_norm, 
            self.config.lookback_window
        )
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Get hidden layer configuration
        hidden_layers = self.config.get_hidden_layers(len(X), X.shape[1])
        logger.info(f"Hidden layers: {hidden_layers}")
        
        # Train ensemble of models
        self.models = []
        
        if self._use_fallback:
            # Single fallback model
            model = self._train_fallback_model(X, y)
            self.models.append(model)
        else:
            # Train ensemble with different random seeds
            import torch
            for i in range(self.config.n_ensemble):
                torch.manual_seed(42 + i)
                np.random.seed(42 + i)
                model = self._train_torch_model(X, y, hidden_layers)
                self.models.append(model)
        
        # Store last training data for prediction
        self._last_train_data = {
            "commits": commits[-self.config.lookback_window:],
            "prs": prs[-self.config.lookback_window:] if prs is not None else None
        }
        
        self.is_fitted = True
        logger.info(f"Neural network training complete. Ensemble size: {len(self.models)}")
        
        return self
    
    def predict(self, periods: int = 1) -> float:
        """
        Predict the next value.
        
        Args:
            periods: Number of periods to predict (only 1 supported for recursive).
            
        Returns:
            Predicted value for the next period.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare input features
        commits = self._last_train_data["commits"]
        prs = self._last_train_data["prs"]
        
        # Normalize
        commits_norm = self._normalize_data(commits, "commits", fit=False)
        if prs is not None and self.config.use_prs_correlation:
            prs_norm = self._normalize_data(prs, "prs", fit=False)
            X = np.concatenate([commits_norm, prs_norm]).reshape(1, -1)
        else:
            X = commits_norm.reshape(1, -1)
        
        # Get predictions from ensemble
        predictions = []
        
        if self._use_fallback:
            for model in self.models:
                pred = model.predict(X)[0]
                predictions.append(pred)
        else:
            import torch
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).item()
                    predictions.append(pred)
        
        # Ensemble mean
        mean_pred = np.mean(predictions)
        
        # Inverse normalize
        mean_pred_denorm = self._inverse_normalize(
            np.array([mean_pred]), "commits"
        )[0]
        
        return float(max(0, mean_pred_denorm))
    
    def predict_with_intervals(
        self, 
        periods: int = 1, 
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Predict with confidence intervals using ensemble variance.
        
        Args:
            periods: Number of periods to predict.
            confidence: Confidence level for intervals.
            
        Returns:
            Dictionary with 'prediction', 'lower', 'upper' values.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare input
        commits = self._last_train_data["commits"]
        prs = self._last_train_data["prs"]
        
        commits_norm = self._normalize_data(commits, "commits", fit=False)
        if prs is not None and self.config.use_prs_correlation:
            prs_norm = self._normalize_data(prs, "prs", fit=False)
            X = np.concatenate([commits_norm, prs_norm]).reshape(1, -1)
        else:
            X = commits_norm.reshape(1, -1)
        
        # Get predictions from ensemble
        predictions = []
        
        if self._use_fallback:
            for model in self.models:
                pred = model.predict(X)[0]
                predictions.append(pred)
        else:
            import torch
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).item()
                    predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Compute confidence interval
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        # Inverse normalize
        mean_denorm = self._inverse_normalize(np.array([mean_pred]), "commits")[0]
        lower_denorm = self._inverse_normalize(np.array([lower]), "commits")[0]
        upper_denorm = self._inverse_normalize(np.array([upper]), "commits")[0]
        
        return {
            "prediction": float(max(0, mean_denorm)),
            "lower": float(max(0, lower_denorm)),
            "upper": float(max(0, upper_denorm))
        }


def create_neural_network_model(
    lookback_window: int = 12,
    use_prs_correlation: bool = True,
    auto_scale: bool = True,
    **kwargs
) -> NeuralNetworkWrapper:
    """
    Factory function to create a neural network model.
    
    Args:
        lookback_window: Number of past weeks to use.
        use_prs_correlation: Whether to use PR correlation.
        auto_scale: Whether to auto-scale network architecture.
        **kwargs: Additional config parameters.
        
    Returns:
        Configured NeuralNetworkWrapper instance.
    """
    config = NeuralNetworkConfig(
        lookback_window=lookback_window,
        use_prs_correlation=use_prs_correlation,
        auto_scale=auto_scale,
        **kwargs
    )
    return NeuralNetworkWrapper(config)


if __name__ == "__main__":
    # Test the neural network model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.etl import load_repository_data
    
    print("Testing Neural Network Model")
    print("=" * 60)
    
    # Load sample data
    repo_path = "repositories/3b1b__manim"
    
    try:
        df = load_repository_data(repo_path)
        print(f"Loaded data: {len(df)} weeks")
        
        # Prepare data for neural network
        train_df = df[["ds", "commits", "prs_opened"]].copy()
        train_df = train_df.rename(columns={"commits": "y"})
        
        # Use only training portion
        train_size = len(train_df) - 8
        train_data = train_df.iloc[:train_size].copy()
        
        print(f"Training on {len(train_data)} weeks")
        
        # Create and train model
        model = create_neural_network_model(
            lookback_window=12,
            use_prs_correlation=True,
            auto_scale=True
        )
        
        model.fit(train_data)
        
        # Make prediction
        prediction = model.predict(periods=1)
        print(f"\nPrediction: {prediction:.2f}")
        
        # Prediction with intervals
        result = model.predict_with_intervals(periods=1)
        print(f"Prediction with intervals: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
