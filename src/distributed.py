"""
Distributed Computing Module - GitHub Activity Predictor
=========================================================

This module provides distributed data processing capabilities
using Dask for scalable ETL and model training.

Features:
- Parallel data loading and transformation
- Distributed model training across workers
- Memory-efficient processing of large datasets
- Integration with existing ETL pipeline
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Check for Dask availability
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster, progress
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available. Install with: pip install 'dask[distributed]'")


class DaskClusterManager:
    """Manages Dask cluster lifecycle."""
    
    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "2GB",
        dashboard_address: str = ":8787"
    ):
        """
        Initialize the cluster manager.
        
        Args:
            n_workers: Number of worker processes.
            threads_per_worker: Threads per worker.
            memory_limit: Memory limit per worker.
            dashboard_address: Dashboard address (e.g., ":8787").
        """
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.dashboard_address = dashboard_address
        
        self.cluster = None
        self.client = None
    
    def start(self) -> "Client":
        """Start the Dask cluster."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask not available")
        
        if self.cluster is not None:
            logger.warning("Cluster already running")
            return self.client
        
        logger.info(f"Starting Dask cluster with {self.n_workers} workers")
        
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            dashboard_address=self.dashboard_address
        )
        
        self.client = Client(self.cluster)
        
        logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
        
        return self.client
    
    def stop(self):
        """Stop the Dask cluster."""
        if self.client:
            self.client.close()
            self.client = None
        
        if self.cluster:
            self.cluster.close()
            self.cluster = None
        
        logger.info("Dask cluster stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class DistributedETL:
    """Distributed ETL pipeline using Dask."""
    
    # File type configuration (from etl.py)
    FILE_CONFIG = {
        "commits.csv": {
            "date_column": "committed_date",
            "metric_name": "commits"
        },
        "issues.csv": {
            "date_column": "created_at",
            "metric_name": "issues_opened"
        },
        "pull_requests.csv": {
            "date_column": "created_at",
            "metric_name": "prs_opened"
        },
        "stargazers.csv": {
            "date_column": "starred_at",
            "metric_name": "new_stars"
        }
    }
    
    def __init__(
        self,
        repositories_path: str = "repositories",
        client: Optional["Client"] = None
    ):
        """
        Initialize distributed ETL.
        
        Args:
            repositories_path: Path to repositories folder.
            client: Optional Dask client (will create if not provided).
        """
        self.repositories_path = Path(repositories_path)
        self.client = client
        self._owns_client = False
        
        if not DASK_AVAILABLE:
            logger.warning("Dask not available - using pandas fallback")
    
    def _ensure_client(self) -> Optional["Client"]:
        """Ensure a Dask client is available."""
        if not DASK_AVAILABLE:
            return None
        
        if self.client is not None:
            return self.client
        
        # Create a default client
        self.client = Client()
        self._owns_client = True
        return self.client
    
    def _cleanup_client(self):
        """Cleanup client if we own it."""
        if self._owns_client and self.client:
            self.client.close()
            self.client = None
            self._owns_client = False
    
    def list_repositories(self) -> List[str]:
        """List all available repositories."""
        if not self.repositories_path.exists():
            return []
        
        repositories = []
        for item in self.repositories_path.iterdir():
            if item.is_dir() and (item / "commits.csv").exists():
                repositories.append(item.name)
        
        return sorted(repositories)
    
    def _process_csv_file(
        self,
        file_path: Path,
        date_column: str,
        metric_name: str
    ) -> pd.DataFrame:
        """Process a single CSV file (used as worker function)."""
        try:
            if not file_path.exists():
                return pd.DataFrame({metric_name: []}, index=pd.DatetimeIndex([], freq="W"))
            
            df = pd.read_csv(file_path)
            
            if date_column not in df.columns:
                return pd.DataFrame({metric_name: []}, index=pd.DatetimeIndex([], freq="W"))
            
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.dropna(subset=[date_column])
            
            df.set_index(date_column, inplace=True)
            weekly = df.resample("W").size()
            weekly.name = metric_name
            
            return weekly.to_frame()
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return pd.DataFrame({metric_name: []}, index=pd.DatetimeIndex([], freq="W"))
    
    def load_repository_distributed(self, repository_name: str) -> pd.DataFrame:
        """
        Load and process repository data using Dask.
        
        Args:
            repository_name: Name of the repository folder.
            
        Returns:
            Weekly aggregated DataFrame.
        """
        repo_path = self.repositories_path / repository_name
        
        if not repo_path.exists():
            raise ValueError(f"Repository not found: {repository_name}")
        
        if DASK_AVAILABLE:
            client = self._ensure_client()
            
            # Submit tasks to workers
            futures = []
            for filename, config in self.FILE_CONFIG.items():
                file_path = repo_path / filename
                future = client.submit(
                    self._process_csv_file,
                    file_path,
                    config["date_column"],
                    config["metric_name"]
                )
                futures.append(future)
            
            # Gather results
            results = client.gather(futures)
            
        else:
            # Fallback to sequential processing
            results = []
            for filename, config in self.FILE_CONFIG.items():
                file_path = repo_path / filename
                result = self._process_csv_file(
                    file_path,
                    config["date_column"],
                    config["metric_name"]
                )
                results.append(result)
        
        # Merge all results
        merged = results[0]
        for df in results[1:]:
            merged = merged.join(df, how="outer")
        
        merged = merged.fillna(0).astype(int)
        merged = merged.reset_index()
        merged.columns = ["ds"] + list(merged.columns[1:])
        
        return merged
    
    def load_all_repositories_distributed(
        self,
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all repositories in parallel.
        
        Args:
            max_workers: Maximum parallel workers.
            
        Returns:
            Dictionary of repository name to DataFrame.
        """
        repositories = self.list_repositories()
        
        if not repositories:
            return {}
        
        if DASK_AVAILABLE:
            client = self._ensure_client()
            
            # Submit all repository loads as tasks
            futures = {
                repo: client.submit(self.load_repository_distributed, repo)
                for repo in repositories
            }
            
            # Gather results
            results = {}
            for repo, future in futures.items():
                try:
                    results[repo] = future.result()
                except Exception as e:
                    logger.error(f"Error loading {repo}: {e}")
            
            return results
        else:
            # Sequential fallback
            return {
                repo: self.load_repository_distributed(repo)
                for repo in repositories
            }
    
    def aggregate_across_repositories(
        self,
        repositories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate data across multiple repositories.
        
        Args:
            repositories: List of repositories to aggregate (all if None).
            
        Returns:
            Aggregated DataFrame.
        """
        if repositories is None:
            repositories = self.list_repositories()
        
        data = self.load_all_repositories_distributed()
        
        if not data:
            return pd.DataFrame()
        
        # Aggregate
        all_dfs = []
        for repo, df in data.items():
            if repo in repositories or repositories is None:
                df["repository"] = repo
                all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Group by date
        aggregated = combined.groupby("ds").agg({
            "commits": "sum",
            "new_stars": "sum",
            "issues_opened": "sum",
            "prs_opened": "sum"
        }).reset_index()
        
        return aggregated


class DistributedModelTrainer:
    """Distributed model training using Dask."""
    
    def __init__(self, client: Optional["Client"] = None):
        """
        Initialize distributed trainer.
        
        Args:
            client: Optional Dask client.
        """
        self.client = client
        self._owns_client = False
    
    def _ensure_client(self) -> Optional["Client"]:
        """Ensure a Dask client is available."""
        if not DASK_AVAILABLE:
            return None
        
        if self.client is not None:
            return self.client
        
        self.client = Client()
        self._owns_client = True
        return self.client
    
    def train_model_task(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: str,
        model_params: Dict
    ) -> Tuple[str, Any, Dict]:
        """
        Train a single model (worker task).
        
        Args:
            data: Training data.
            target_column: Target column name.
            model_type: Type of model.
            model_params: Model parameters.
            
        Returns:
            Tuple of (model_type, trained_model, metrics).
        """
        from src.model_engine import ProphetWrapper, FARIMAWrapper
        
        try:
            if model_type == "prophet":
                model = ProphetWrapper(**model_params)
            elif model_type == "farima":
                model = FARIMAWrapper(**model_params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train
            model.fit(data, target_col=target_column)
            
            # Get in-sample predictions for evaluation
            predictions = model.predict(len(data))
            
            # Calculate metrics
            actual = data[target_column].values
            predicted = predictions["yhat"].values[:len(actual)]
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "n_samples": len(data)
            }
            
            return model_type, model, metrics
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            return model_type, None, {"error": str(e)}
    
    def train_multiple_models_distributed(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_configs: List[Dict]
    ) -> Dict[str, Tuple[Any, Dict]]:
        """
        Train multiple models in parallel.
        
        Args:
            data: Training data.
            target_column: Target column.
            model_configs: List of {"type": ..., "params": {...}}.
            
        Returns:
            Dict of model_type to (model, metrics).
        """
        if DASK_AVAILABLE:
            client = self._ensure_client()
            
            # Submit training tasks
            futures = []
            for config in model_configs:
                future = client.submit(
                    self.train_model_task,
                    data,
                    target_column,
                    config["type"],
                    config.get("params", {})
                )
                futures.append(future)
            
            # Gather results
            results_list = client.gather(futures)
            
        else:
            # Sequential fallback
            results_list = [
                self.train_model_task(
                    data,
                    target_column,
                    config["type"],
                    config.get("params", {})
                )
                for config in model_configs
            ]
        
        results = {}
        for model_type, model, metrics in results_list:
            results[model_type] = (model, metrics)
        
        return results
    
    def hyperparameter_search_distributed(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: str,
        param_grid: Dict[str, List]
    ) -> Tuple[Dict, float]:
        """
        Distributed hyperparameter search.
        
        Args:
            data: Training data.
            target_column: Target column.
            model_type: Model type.
            param_grid: Parameter grid.
            
        Returns:
            Tuple of (best_params, best_score).
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        configs = []
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            configs.append({"type": model_type, "params": params})
        
        logger.info(f"Running distributed search over {len(configs)} configurations")
        
        # Train all configurations
        results = self.train_multiple_models_distributed(
            data, target_column, configs
        )
        
        # Find best
        best_params = None
        best_score = float("inf")
        
        for config in configs:
            model_key = f"{model_type}_{hash(str(config['params']))}"
            if model_key in results:
                _, metrics = results[model_key]
                if "mae" in metrics and metrics["mae"] < best_score:
                    best_score = metrics["mae"]
                    best_params = config["params"]
        
        if best_params is None and configs:
            best_params = configs[0]["params"]
        
        return best_params, best_score


class DistributedProcessor:
    """High-level interface for distributed processing."""
    
    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "2GB"
    ):
        """
        Initialize distributed processor.
        
        Args:
            n_workers: Number of workers.
            threads_per_worker: Threads per worker.
            memory_limit: Memory limit per worker.
        """
        self.cluster_manager = DaskClusterManager(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit
        )
        
        self.etl = None
        self.trainer = None
        self._running = False
    
    def start(self):
        """Start the distributed processing cluster."""
        if not DASK_AVAILABLE:
            logger.warning("Dask not available - running in local mode")
            self._running = True
            return
        
        client = self.cluster_manager.start()
        self.etl = DistributedETL(client=client)
        self.trainer = DistributedModelTrainer(client=client)
        self._running = True
        
        logger.info("Distributed processor started")
    
    def stop(self):
        """Stop the distributed processing cluster."""
        self.cluster_manager.stop()
        self._running = False
        logger.info("Distributed processor stopped")
    
    def process_repository(self, repository_name: str) -> pd.DataFrame:
        """Process a single repository."""
        if not self._running:
            self.start()
        
        if self.etl is None:
            self.etl = DistributedETL()
        
        return self.etl.load_repository_distributed(repository_name)
    
    def process_all_repositories(self) -> Dict[str, pd.DataFrame]:
        """Process all repositories."""
        if not self._running:
            self.start()
        
        if self.etl is None:
            self.etl = DistributedETL()
        
        return self.etl.load_all_repositories_distributed()
    
    def train_models(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_types: List[str] = ["prophet", "farima"]
    ) -> Dict[str, Tuple[Any, Dict]]:
        """Train multiple models on the data."""
        if not self._running:
            self.start()
        
        if self.trainer is None:
            self.trainer = DistributedModelTrainer()
        
        configs = [{"type": mt, "params": {}} for mt in model_types]
        return self.trainer.train_multiple_models_distributed(
            data, target_column, configs
        )
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def create_distributed_processor(
    n_workers: int = 4,
    memory_limit: str = "2GB"
) -> DistributedProcessor:
    """Factory function to create a distributed processor."""
    return DistributedProcessor(
        n_workers=n_workers,
        memory_limit=memory_limit
    )


if __name__ == "__main__":
    print("Testing Distributed Computing Module")
    print("=" * 60)
    
    if not DASK_AVAILABLE:
        print("Dask not available. Install with: pip install 'dask[distributed]'")
        print("Running in local mode...")
    
    # Create processor
    processor = create_distributed_processor(n_workers=2)
    
    try:
        processor.start()
        
        # List repositories
        if processor.etl:
            repos = processor.etl.list_repositories()
            print(f"\nFound {len(repos)} repositories")
            
            if repos:
                # Process first repository
                repo = repos[0]
                print(f"\nProcessing: {repo}")
                
                data = processor.process_repository(repo)
                print(f"Loaded data shape: {data.shape}")
                print(data.head())
        
    finally:
        processor.stop()
    
    print("\nDistributed processing test completed")
