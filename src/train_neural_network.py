"""
Neural Network Training Script - GitHub Activity Predictor
===========================================================

This script trains a neural network model for predicting GitHub commits,
evaluates performance with R² and MSE metrics, plots training losses,
and saves model checkpoints for real-time inference.

Usage:
    python -m src.train_neural_network --repository <repo_name>
    python -m src.train_neural_network --all
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl import load_repository_data, list_available_repositories
from src.neural_network import NeuralNetworkConfig, NeuralNetworkWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directory for saving checkpoints and plots
CHECKPOINT_DIR = Path("checkpoints")
PLOTS_DIR = Path("training_plots")
METRICS_DIR = Path("training_metrics")


class NeuralNetworkTrainer:
    """
    Trainer class for neural network model with comprehensive tracking.
    """
    
    def __init__(
        self,
        config: Optional[NeuralNetworkConfig] = None,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        plots_dir: Path = PLOTS_DIR
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Neural network configuration.
            checkpoint_dir: Directory for saving checkpoints.
            plots_dir: Directory for saving plots.
        """
        self.config = config or NeuralNetworkConfig(
            lookback_window=24,  # Increased lookback for more context
            epochs=300,
            learning_rate=0.0005,  # Reduced LR for stability
            batch_size=16,
            early_stopping_patience=30,
            n_ensemble=5,
            use_prs_correlation=True,
            dropout_rate=0.4,  # Increased dropout
            use_feature_engineering=True,  # Enable feature engineering
            l2_regularization=0.01,  # L2 regularization
            use_batch_norm=True,
            use_residual=True,
            noise_injection=0.05
        )
        
        self.checkpoint_dir = checkpoint_dir
        self.plots_dir = plots_dir
        self.metrics_dir = METRICS_DIR
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_r2": [],
            "val_r2": []
        }
        
        self.model = None
        self.best_model_state = None
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "commits",
        test_size: float = 0.15  # Reduced test size for more training data
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training and validation.
        
        Args:
            df: Input DataFrame with time series data.
            target_col: Target column name.
            test_size: Fraction of data for validation.
            
        Returns:
            Tuple of (train_df, val_df).
        """
        # Ensure required columns exist
        if "ds" not in df.columns:
            if df.index.name == "ds":
                df = df.reset_index()
            else:
                raise ValueError("DataFrame must have 'ds' column")
        
        # Rename target to 'y' for consistency
        if target_col != "y":
            df = df.rename(columns={target_col: "y"})
        
        # Sort by date
        df = df.sort_values("ds").reset_index(drop=True)
        
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}")
        
        return train_df, val_df
    
    def train_with_tracking(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        repository_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Train the model with comprehensive metric tracking.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            repository_name: Name of the repository for labeling.
            
        Returns:
            Dictionary with training results and metrics.
        """
        logger.info(f"Starting training for repository: {repository_name}")
        
        # Reset history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_r2": [],
            "val_r2": [],
            "epochs": []
        }
        
        # Create model
        self.model = NeuralNetworkWrapper(self.config)
        
        # Check if PyTorch is available for detailed tracking
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            use_torch = True
        except ImportError:
            use_torch = False
            logger.warning("PyTorch not available - using sklearn fallback (limited tracking)")
        
        # Extract data
        train_commits = train_df["y"].values.astype(float)
        val_commits = np.concatenate([train_df["y"].values, val_df["y"].values]).astype(float)
        
        train_prs = None
        val_prs = None
        if "prs_opened" in train_df.columns and self.config.use_prs_correlation:
            train_prs = train_df["prs_opened"].values.astype(float)
            val_prs = np.concatenate([train_df["prs_opened"].values, val_df["prs_opened"].values]).astype(float)
        
        if use_torch:
            # Detailed PyTorch training with epoch-level tracking
            results = self._train_torch_with_tracking(
                train_commits, val_commits,
                train_prs, val_prs,
                len(train_df), len(val_df),
                repository_name
            )
        else:
            # Fallback training
            results = self._train_fallback_with_tracking(
                train_df, val_df, repository_name
            )
        
        return results
    
    def _train_torch_with_tracking(
        self,
        train_commits: np.ndarray,
        all_commits: np.ndarray,
        train_prs: Optional[np.ndarray],
        all_prs: Optional[np.ndarray],
        train_len: int,
        val_len: int,
        repository_name: str
    ) -> Dict[str, Any]:
        """Train with PyTorch and track metrics per epoch."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import MinMaxScaler
        
        # Normalize data
        commit_scaler = MinMaxScaler()
        train_commits_norm = commit_scaler.fit_transform(
            train_commits.reshape(-1, 1)
        ).flatten()
        
        all_commits_norm = commit_scaler.transform(
            all_commits.reshape(-1, 1)
        ).flatten()
        
        prs_scaler = None
        train_prs_norm = None
        all_prs_norm = None
        
        if train_prs is not None:
            prs_scaler = MinMaxScaler()
            train_prs_norm = prs_scaler.fit_transform(
                train_prs.reshape(-1, 1)
            ).flatten()
            all_prs_norm = prs_scaler.transform(
                all_prs.reshape(-1, 1)
            ).flatten()
        
        # Create sequences
        lookback = self.config.lookback_window
        
        def create_sequences(commits, prs, length):
            n_samples = length - lookback
            if prs is not None:
                X = np.zeros((n_samples, lookback * 2))
                for i in range(n_samples):
                    X[i, :lookback] = commits[i:i + lookback]
                    X[i, lookback:] = prs[i:i + lookback]
            else:
                X = np.zeros((n_samples, lookback))
                for i in range(n_samples):
                    X[i] = commits[i:i + lookback]
            y = commits[lookback:length]
            return X, y
        
        X_train, y_train = create_sequences(
            train_commits_norm, train_prs_norm, len(train_commits_norm)
        )
        
        # Validation: use the continuation of training data
        X_val, y_val = create_sequences(
            all_commits_norm[train_len - lookback:],
            all_prs_norm[train_len - lookback:] if all_prs_norm is not None else None,
            val_len + lookback
        )
        
        logger.info(f"Training sequences: X_train={X_train.shape}, X_val={X_val.shape}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(X_train)),
            shuffle=True
        )
        
        # Build model
        hidden_layers = self.config.get_hidden_layers(len(X_train), X_train.shape[1])
        logger.info(f"Model architecture: input={X_train.shape[1]} -> {hidden_layers} -> 1")
        
        # Build PyTorch model
        layers = []
        input_size = X_train.shape[1]
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        model = nn.Sequential(*layers)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)
                train_preds.extend(outputs.detach().numpy().flatten())
                train_targets.extend(batch_y.numpy().flatten())
            
            train_loss /= len(X_train)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = val_outputs.numpy().flatten()
            
            # Calculate metrics (in normalized space)
            train_mse = mean_squared_error(train_targets, train_preds)
            val_mse = mean_squared_error(y_val, val_preds)
            
            # R² scores
            train_r2 = r2_score(train_targets, train_preds) if len(train_targets) > 1 else 0.0
            val_r2 = r2_score(y_val, val_preds) if len(y_val) > 1 else 0.0
            
            # Store history
            self.training_history["epochs"].append(epoch + 1)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_mse"].append(train_mse)
            self.training_history["val_mse"].append(val_mse)
            self.training_history["train_r2"].append(train_r2)
            self.training_history["val_r2"].append(val_r2)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}"
                )
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_train_preds = model(X_train_t).numpy().flatten()
            final_val_preds = model(X_val_t).numpy().flatten()
        
        # Inverse transform predictions for actual metrics
        train_preds_actual = commit_scaler.inverse_transform(
            final_train_preds.reshape(-1, 1)
        ).flatten()
        train_targets_actual = commit_scaler.inverse_transform(
            y_train.reshape(-1, 1)
        ).flatten()
        
        val_preds_actual = commit_scaler.inverse_transform(
            final_val_preds.reshape(-1, 1)
        ).flatten()
        val_targets_actual = commit_scaler.inverse_transform(
            y_val.reshape(-1, 1)
        ).flatten()
        
        # Calculate final metrics in actual scale
        final_train_mse = mean_squared_error(train_targets_actual, train_preds_actual)
        final_val_mse = mean_squared_error(val_targets_actual, val_preds_actual)
        final_train_r2 = r2_score(train_targets_actual, train_preds_actual)
        final_val_r2 = r2_score(val_targets_actual, val_preds_actual)
        final_train_rmse = np.sqrt(final_train_mse)
        final_val_rmse = np.sqrt(final_val_mse)
        
        # Store model and scalers for checkpoint
        self.model_state = {
            "model_state_dict": model.state_dict(),
            "model_architecture": {
                "input_size": X_train.shape[1],
                "hidden_layers": hidden_layers,
                "dropout_rate": self.config.dropout_rate
            },
            "scalers": {
                "commits": commit_scaler,
                "prs": prs_scaler
            },
            "config": self.config.to_dict(),
            "lookback_window": self.config.lookback_window,
            "use_prs": train_prs is not None
        }
        
        results = {
            "repository": repository_name,
            "epochs_trained": len(self.training_history["epochs"]),
            "best_epoch": self.training_history["val_loss"].index(min(self.training_history["val_loss"])) + 1,
            "metrics": {
                "train_mse": final_train_mse,
                "val_mse": final_val_mse,
                "train_rmse": final_train_rmse,
                "val_rmse": final_val_rmse,
                "train_r2": final_train_r2,
                "val_r2": final_val_r2
            },
            "predictions": {
                "train_actual": train_targets_actual.tolist(),
                "train_predicted": train_preds_actual.tolist(),
                "val_actual": val_targets_actual.tolist(),
                "val_predicted": val_preds_actual.tolist()
            },
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete for {repository_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Final Metrics:")
        logger.info(f"  Train MSE: {final_train_mse:.4f} | Val MSE: {final_val_mse:.4f}")
        logger.info(f"  Train RMSE: {final_train_rmse:.4f} | Val RMSE: {final_val_rmse:.4f}")
        logger.info(f"  Train R²: {final_train_r2:.4f} | Val R²: {final_val_r2:.4f}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _train_fallback_with_tracking(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        repository_name: str
    ) -> Dict[str, Any]:
        """Improved fallback training with ensemble and feature engineering."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        # Prepare data
        train_commits = train_df["y"].values.astype(float)
        val_commits = val_df["y"].values.astype(float)
        
        # Combine for full sequence
        all_commits = np.concatenate([train_commits, val_commits])
        
        # Normalize
        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(train_commits.reshape(-1, 1)).flatten()
        all_norm = scaler.transform(all_commits.reshape(-1, 1)).flatten()
        
        # Create sequences with feature engineering
        lookback = self.config.lookback_window
        
        def compute_features(data, idx, lookback):
            """Compute enhanced features for a single sample."""
            window = data[idx:idx + lookback]
            
            features = list(window)  # Raw lookback values
            
            # Rolling statistics at multiple scales
            features.extend([
                np.mean(window),
                np.std(window) + 1e-8,
                np.min(window),
                np.max(window),
                np.median(window),
                window[-1] - window[0],  # Overall trend
            ])
            
            # Multi-scale means
            if lookback >= 4:
                features.append(np.mean(window[-4:]))  # Last 4 weeks
                features.append(np.mean(window[:4]))   # First 4 weeks
            if lookback >= 8:
                features.append(np.mean(window[-8:]))  # Last 8 weeks
            if lookback >= 12:
                features.append(np.mean(window[-12:]))  # Last 12 weeks
            
            # Momentum and rate of change
            if lookback >= 2:
                features.append(window[-1] - window[-2])  # 1-week change
            if lookback >= 4:
                features.append(window[-1] - window[-4])  # 4-week change
                features.append(np.mean(window[-2:]) - np.mean(window[-4:-2]))  # Momentum
            if lookback >= 8:
                features.append(window[-1] - window[-8])  # 8-week change
            
            # Volatility measures
            if lookback >= 4:
                features.append(np.std(window[-4:]) / (np.mean(window[-4:]) + 1e-8))  # CV 4w
            if lookback >= 8:
                features.append(np.std(window[-8:]) / (np.mean(window[-8:]) + 1e-8))  # CV 8w
            
            # Percentiles
            features.append(np.percentile(window, 25))
            features.append(np.percentile(window, 75))
            
            # Trend strength (linear regression slope)
            x = np.arange(len(window))
            if len(window) > 1:
                slope = np.polyfit(x, window, 1)[0]
                features.append(slope)
            else:
                features.append(0)
            
            return np.array(features)
        
        def create_enhanced_sequences(data, start_idx, end_idx):
            X, y = [], []
            for i in range(start_idx, end_idx - lookback):
                features = compute_features(data, i, lookback)
                X.append(features)
                y.append(data[i + lookback])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_enhanced_sequences(train_norm, 0, len(train_norm))
        X_val, y_val = create_enhanced_sequences(all_norm, len(train_norm) - lookback, len(all_norm))
        
        logger.info(f"Enhanced features: X_train shape = {X_train.shape}")
        
        # Standardize features
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        
        # Add noise for robustness
        noise = np.random.normal(0, 0.02, X_train_scaled.shape)
        X_train_noisy = X_train_scaled + noise
        
        # Train model with improved architecture
        hidden_layers = self.config.get_hidden_layers(len(X_train), X_train.shape[1])
        
        # Gradient Boosting - primary model for time series
        gb_model = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_split=8,
            min_samples_leaf=4,
            learning_rate=0.015,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Second GB - different hyperparams for diversity
        gb_model2 = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=3,
            learning_rate=0.03,
            subsample=0.7,
            random_state=123
        )
        
        # Neural network - captures nonlinear patterns
        nn_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
            verbose=True
        )
        
        # Ensemble - balanced mix
        ensemble = VotingRegressor([
            ('gb1', gb_model),
            ('gb2', gb_model2),
            ('nn', nn_model)
        ])
        
        logger.info(f"Training ensemble with hidden layers: {hidden_layers}")
        ensemble.fit(X_train_noisy, y_train)
        
        # Get loss curve from NN component
        if hasattr(nn_model, 'loss_curve_'):
            self.training_history["train_loss"] = nn_model.loss_curve_
            self.training_history["epochs"] = list(range(1, len(nn_model.loss_curve_) + 1))
        
        # Predictions
        train_preds = ensemble.predict(X_train_scaled)
        val_preds = ensemble.predict(X_val_scaled)
        
        # Inverse transform
        train_preds_actual = scaler.inverse_transform(train_preds.reshape(-1, 1)).flatten()
        train_targets_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        val_preds_actual = scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
        val_targets_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        final_train_mse = mean_squared_error(train_targets_actual, train_preds_actual)
        final_val_mse = mean_squared_error(val_targets_actual, val_preds_actual)
        final_train_r2 = r2_score(train_targets_actual, train_preds_actual)
        final_val_r2 = r2_score(val_targets_actual, val_preds_actual)
        
        # Store model for checkpoint
        self.model_state = {
            "model": ensemble,
            "scaler": scaler,
            "feature_scaler": feature_scaler,
            "config": self.config.to_dict(),
            "lookback_window": self.config.lookback_window,
            "type": "sklearn_ensemble"
        }
        
        results = {
            "repository": repository_name,
            "epochs_trained": nn_model.n_iter_ if hasattr(nn_model, 'n_iter_') else 0,
            "metrics": {
                "train_mse": final_train_mse,
                "val_mse": final_val_mse,
                "train_rmse": np.sqrt(final_train_mse),
                "val_rmse": np.sqrt(final_val_mse),
                "train_r2": final_train_r2,
                "val_r2": final_val_r2
            },
            "predictions": {
                "train_actual": train_targets_actual.tolist(),
                "train_predicted": train_preds_actual.tolist(),
                "val_actual": val_targets_actual.tolist(),
                "val_predicted": val_preds_actual.tolist()
            },
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"\nTraining Complete - Val MSE: {final_val_mse:.4f}, Val R²: {final_val_r2:.4f}")
        
        return results
    
    def plot_training_curves(
        self,
        results: Dict[str, Any],
        repository_name: str,
        save: bool = True,
        show: bool = True
    ) -> str:
        """
        Plot training curves and save to file.
        
        Args:
            results: Training results dictionary.
            repository_name: Repository name for title.
            save: Whether to save the plot.
            show: Whether to display the plot.
            
        Returns:
            Path to saved plot file.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Neural Network Training Results - {repository_name}', fontsize=14, fontweight='bold')
        
        history = results.get("training_history", self.training_history)
        epochs = history.get("epochs", list(range(1, len(history.get("train_loss", [])) + 1)))
        
        # Plot 1: Training and Validation Loss
        ax1 = axes[0, 0]
        if history.get("train_loss"):
            ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        if history.get("val_loss"):
            ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² Score
        ax2 = axes[0, 1]
        if history.get("train_r2"):
            ax2.plot(epochs, history["train_r2"], 'b-', label='Train R²', linewidth=2)
        if history.get("val_r2"):
            ax2.plot(epochs, history["val_r2"], 'r-', label='Val R²', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actual vs Predicted (Training)
        ax3 = axes[1, 0]
        predictions = results.get("predictions", {})
        if predictions.get("train_actual") and predictions.get("train_predicted"):
            train_actual = predictions["train_actual"]
            train_pred = predictions["train_predicted"]
            ax3.scatter(train_actual, train_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
            max_val = max(max(train_actual), max(train_pred))
            ax3.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            ax3.set_xlabel('Actual Commits')
            ax3.set_ylabel('Predicted Commits')
            ax3.set_title(f'Train: Actual vs Predicted (R²={results["metrics"]["train_r2"]:.4f})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Actual vs Predicted (Validation)
        ax4 = axes[1, 1]
        if predictions.get("val_actual") and predictions.get("val_predicted"):
            val_actual = predictions["val_actual"]
            val_pred = predictions["val_predicted"]
            ax4.scatter(val_actual, val_pred, alpha=0.6, color='orange', edgecolors='black', linewidth=0.5)
            max_val = max(max(val_actual), max(val_pred))
            ax4.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            ax4.set_xlabel('Actual Commits')
            ax4.set_ylabel('Predicted Commits')
            ax4.set_title(f'Validation: Actual vs Predicted (R²={results["metrics"]["val_r2"]:.4f})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = ""
        if save:
            safe_name = repository_name.replace("/", "_").replace("__", "_")
            plot_path = self.plots_dir / f"training_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training plot saved to: {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(plot_path)
    
    def plot_predictions_timeline(
        self,
        results: Dict[str, Any],
        repository_name: str,
        save: bool = True,
        show: bool = True
    ) -> str:
        """
        Plot predictions as a timeline.
        
        Args:
            results: Training results dictionary.
            repository_name: Repository name for title.
            save: Whether to save the plot.
            show: Whether to display the plot.
            
        Returns:
            Path to saved plot file.
        """
        predictions = results.get("predictions", {})
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        train_actual = predictions.get("train_actual", [])
        train_pred = predictions.get("train_predicted", [])
        val_actual = predictions.get("val_actual", [])
        val_pred = predictions.get("val_predicted", [])
        
        # Create timeline
        train_x = list(range(len(train_actual)))
        val_x = list(range(len(train_actual), len(train_actual) + len(val_actual)))
        
        # Plot actual values
        ax.plot(train_x, train_actual, 'b-', label='Train Actual', linewidth=1.5, alpha=0.8)
        ax.plot(val_x, val_actual, 'g-', label='Val Actual', linewidth=1.5, alpha=0.8)
        
        # Plot predictions
        ax.plot(train_x, train_pred, 'b--', label='Train Predicted', linewidth=1.5, alpha=0.6)
        ax.plot(val_x, val_pred, 'r--', label='Val Predicted', linewidth=2)
        
        # Mark train/val split
        ax.axvline(x=len(train_actual), color='gray', linestyle=':', linewidth=2, label='Train/Val Split')
        
        ax.set_xlabel('Week Index')
        ax.set_ylabel('Number of Commits')
        ax.set_title(f'Commits Prediction Timeline - {repository_name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = ""
        if save:
            safe_name = repository_name.replace("/", "_").replace("__", "_")
            plot_path = self.plots_dir / f"timeline_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Timeline plot saved to: {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(plot_path)
    
    def save_checkpoint(
        self,
        results: Dict[str, Any],
        repository_name: str
    ) -> str:
        """
        Save model checkpoint for inference.
        
        Args:
            results: Training results.
            repository_name: Repository name.
            
        Returns:
            Path to checkpoint file.
        """
        safe_name = repository_name.replace("/", "_").replace("__", "_")
        checkpoint_path = self.checkpoint_dir / f"nn_model_{safe_name}.pkl"
        
        checkpoint = {
            "model_state": self.model_state,
            "results": results,
            "repository": repository_name,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        # Also save a "latest" checkpoint for easy access
        latest_path = self.checkpoint_dir / "nn_model_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save metrics as JSON
        metrics_path = self.metrics_dir / f"metrics_{safe_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "repository": repository_name,
                "metrics": results["metrics"],
                "epochs_trained": results["epochs_trained"],
                "timestamp": results["timestamp"]
            }, f, indent=2)
        
        return str(checkpoint_path)
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint dictionary.
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint


def train_repository(
    repository_name: str,
    data_dir: str = "repositories",
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Train neural network on a single repository.
    
    Args:
        repository_name: Name of the repository folder.
        data_dir: Directory containing repository data.
        show_plots: Whether to display plots.
        
    Returns:
        Training results dictionary.
    """
    # Load data
    repo_path = Path(data_dir) / repository_name
    logger.info(f"Loading data from: {repo_path}")
    
    df = load_repository_data(str(repo_path))
    logger.info(f"Loaded {len(df)} weeks of data")
    
    # Initialize trainer
    config = NeuralNetworkConfig(
        lookback_window=12,
        epochs=200,
        learning_rate=0.001,
        batch_size=32,
        early_stopping_patience=15,
        n_ensemble=5,
        use_prs_correlation=True,
        dropout_rate=0.2
    )
    
    trainer = NeuralNetworkTrainer(config)
    
    # Prepare data
    train_df, val_df = trainer.prepare_data(df, target_col="commits", test_size=0.2)
    
    # Train
    results = trainer.train_with_tracking(train_df, val_df, repository_name)
    
    # Plot results
    trainer.plot_training_curves(results, repository_name, save=True, show=show_plots)
    trainer.plot_predictions_timeline(results, repository_name, save=True, show=show_plots)
    
    # Save checkpoint
    checkpoint_path = trainer.save_checkpoint(results, repository_name)
    results["checkpoint_path"] = checkpoint_path
    
    return results


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train Neural Network for GitHub Commit Prediction")
    parser.add_argument("--repository", "-r", type=str, help="Repository name to train on")
    parser.add_argument("--all", "-a", action="store_true", help="Train on all repositories")
    parser.add_argument("--data-dir", "-d", type=str, default="repositories", help="Data directory")
    parser.add_argument("--no-plots", action="store_true", help="Don't display plots")
    parser.add_argument("--list", "-l", action="store_true", help="List available repositories")
    
    args = parser.parse_args()
    
    if args.list:
        repos = list_available_repositories(args.data_dir)
        print(f"\nAvailable repositories ({len(repos)}):")
        for repo in repos[:20]:
            print(f"  - {repo}")
        if len(repos) > 20:
            print(f"  ... and {len(repos) - 20} more")
        return
    
    if args.all:
        repos = list_available_repositories(args.data_dir)
        logger.info(f"Training on {len(repos)} repositories...")
        
        all_results = []
        for repo in repos[:5]:  # Limit to first 5 for demo
            try:
                results = train_repository(repo, args.data_dir, show_plots=not args.no_plots)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error training {repo}: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for r in all_results:
            print(f"{r['repository']:40} | Val R²: {r['metrics']['val_r2']:.4f} | Val MSE: {r['metrics']['val_mse']:.2f}")
        
    elif args.repository:
        results = train_repository(args.repository, args.data_dir, show_plots=not args.no_plots)
        
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Repository: {results['repository']}")
        print(f"Epochs Trained: {results['epochs_trained']}")
        print(f"Train MSE: {results['metrics']['train_mse']:.4f}")
        print(f"Val MSE: {results['metrics']['val_mse']:.4f}")
        print(f"Train R²: {results['metrics']['train_r2']:.4f}")
        print(f"Val R²: {results['metrics']['val_r2']:.4f}")
        print(f"Checkpoint: {results.get('checkpoint_path', 'N/A')}")
        
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.train_neural_network --list")
        print("  python -m src.train_neural_network -r tensorflow__tensorflow")
        print("  python -m src.train_neural_network --all --no-plots")


if __name__ == "__main__":
    main()
