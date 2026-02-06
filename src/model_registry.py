"""
Model Registry Module - GitHub Activity Predictor
==================================================

This module implements model versioning and a formal model registry
to track experiment results, model versions, lifecycle stages, and
deployment metadata.

Features:
- Model versioning with semantic versioning
- Lifecycle stage management (staging, production, archived)
- Experiment result tracking
- Deployment metadata storage
- MLflow integration for model registry
- Model comparison and selection
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import pickle

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    rmse: float
    mae: float
    smape: float
    r2_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_better_than(self, other: "ModelMetrics", metric: str = "smape") -> bool:
        """Check if this model is better than another based on a metric."""
        self_val = getattr(self, metric)
        other_val = getattr(other, metric)
        
        # Lower is better for all our metrics
        return self_val < other_val


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    
    version: str
    model_type: str
    target_column: str
    repository_name: str
    metrics: ModelMetrics
    parameters: Dict
    stage: ModelStage = ModelStage.DEVELOPMENT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    tags: List[str] = field(default_factory=list)
    artifact_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    deployment_metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        data["stage"] = ModelStage(data["stage"])
        data["metrics"] = ModelMetrics(**data["metrics"])
        return cls(**data)


@dataclass 
class ExperimentResult:
    """Result from a model training experiment."""
    
    experiment_id: str
    model_type: str
    target_column: str
    repository_name: str
    metrics: ModelMetrics
    parameters: Dict
    predictions: List[Dict]
    training_duration_seconds: float
    data_size: int
    horizon: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    mlflow_run_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LocalModelRegistry:
    """
    Local model registry for tracking model versions.
    Can work standalone or alongside MLflow.
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to store registry data.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.registry_path / "models.json"
        self.experiments_file = self.registry_path / "experiments.json"
        
        self.models = self._load_models()
        self.experiments = self._load_experiments()
        
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _load_models(self) -> Dict[str, List[ModelVersion]]:
        """Load models from registry file."""
        if self.models_file.exists():
            try:
                with open(self.models_file, "r") as f:
                    data = json.load(f)
                    models = {}
                    for model_name, versions in data.items():
                        models[model_name] = [
                            ModelVersion.from_dict(v) for v in versions
                        ]
                    return models
            except Exception as e:
                logger.warning(f"Error loading models: {e}")
        return {}
    
    def _load_experiments(self) -> List[ExperimentResult]:
        """Load experiments from file."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    data = json.load(f)
                    return [
                        ExperimentResult(
                            **{**exp, "metrics": ModelMetrics(**exp["metrics"])}
                        ) for exp in data
                    ]
            except Exception as e:
                logger.warning(f"Error loading experiments: {e}")
        return []
    
    def _save_models(self):
        """Save models to registry file."""
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [v.to_dict() for v in versions]
        
        with open(self.models_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _save_experiments(self):
        """Save experiments to file."""
        data = [exp.to_dict() for exp in self.experiments]
        with open(self.experiments_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_model_name(
        self, 
        model_type: str, 
        target_column: str, 
        repository_name: str
    ) -> str:
        """Generate a unique model name."""
        return f"{repository_name}_{target_column}_{model_type}"
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model."""
        if model_name not in self.models or not self.models[model_name]:
            return "1.0.0"
        
        versions = self.models[model_name]
        latest = max(versions, key=lambda v: v.created_at)
        
        # Parse version and increment patch
        parts = latest.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor}.{patch + 1}"
    
    def register_model(
        self,
        model_type: str,
        target_column: str,
        repository_name: str,
        metrics: ModelMetrics,
        parameters: Dict,
        artifact_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_type: Type of model (prophet, farima, neural_network).
            target_column: Target column being predicted.
            repository_name: Repository the model is trained on.
            metrics: Model performance metrics.
            parameters: Model hyperparameters.
            artifact_path: Path to saved model artifact.
            mlflow_run_id: MLflow run ID if applicable.
            description: Model description.
            tags: Model tags.
            
        Returns:
            Registered ModelVersion.
        """
        model_name = self._generate_model_name(model_type, target_column, repository_name)
        version = self._generate_version(model_name)
        
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            target_column=target_column,
            repository_name=repository_name,
            metrics=metrics,
            parameters=parameters,
            artifact_path=artifact_path,
            mlflow_run_id=mlflow_run_id,
            description=description,
            tags=tags or []
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        self.models[model_name].append(model_version)
        self._save_models()
        
        logger.info(f"Registered model {model_name} version {version}")
        
        return model_version
    
    def update_stage(
        self, 
        model_name: str, 
        version: str, 
        new_stage: ModelStage
    ) -> bool:
        """
        Update the stage of a model version.
        
        Args:
            model_name: Name of the model.
            version: Version to update.
            new_stage: New lifecycle stage.
            
        Returns:
            True if successful.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        for model_version in self.models[model_name]:
            if model_version.version == version:
                old_stage = model_version.stage
                model_version.stage = new_stage
                model_version.updated_at = datetime.utcnow().isoformat()
                self._save_models()
                
                logger.info(
                    f"Updated {model_name} v{version}: "
                    f"{old_stage.value} -> {new_stage.value}"
                )
                return True
        
        logger.error(f"Version {version} not found for {model_name}")
        return False
    
    def promote_to_production(
        self, 
        model_name: str, 
        version: str,
        archive_current: bool = True
    ) -> bool:
        """
        Promote a model version to production.
        
        Args:
            model_name: Name of the model.
            version: Version to promote.
            archive_current: Whether to archive the current production model.
            
        Returns:
            True if successful.
        """
        if model_name not in self.models:
            return False
        
        # Archive current production model
        if archive_current:
            for mv in self.models[model_name]:
                if mv.stage == ModelStage.PRODUCTION:
                    self.update_stage(model_name, mv.version, ModelStage.ARCHIVED)
        
        return self.update_stage(model_name, version, ModelStage.PRODUCTION)
    
    def get_production_model(
        self, 
        model_name: str
    ) -> Optional[ModelVersion]:
        """Get the current production model."""
        if model_name not in self.models:
            return None
        
        for mv in self.models[model_name]:
            if mv.stage == ModelStage.PRODUCTION:
                return mv
        
        return None
    
    def get_latest_version(
        self, 
        model_name: str,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """Get the latest version of a model, optionally filtered by stage."""
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if not versions:
            return None
        
        return max(versions, key=lambda v: v.created_at)
    
    def get_best_model(
        self, 
        model_name: str,
        metric: str = "smape"
    ) -> Optional[ModelVersion]:
        """Get the best performing model based on a metric."""
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        if not versions:
            return None
        
        return min(versions, key=lambda v: getattr(v.metrics, metric))
    
    def compare_models(
        self,
        model_names: List[str],
        metric: str = "smape"
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare.
            metric: Metric to compare on.
            
        Returns:
            DataFrame with comparison.
        """
        rows = []
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
            
            for mv in self.models[model_name]:
                rows.append({
                    "model_name": model_name,
                    "version": mv.version,
                    "model_type": mv.model_type,
                    "stage": mv.stage.value,
                    "rmse": mv.metrics.rmse,
                    "mae": mv.metrics.mae,
                    "smape": mv.metrics.smape,
                    "created_at": mv.created_at
                })
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(metric)
        
        return df
    
    def log_experiment(
        self,
        model_type: str,
        target_column: str,
        repository_name: str,
        metrics: ModelMetrics,
        parameters: Dict,
        predictions: List[Dict],
        training_duration: float,
        data_size: int,
        horizon: int,
        mlflow_run_id: Optional[str] = None
    ) -> ExperimentResult:
        """
        Log an experiment result.
        
        Args:
            model_type: Type of model used.
            target_column: Target column predicted.
            repository_name: Repository used.
            metrics: Resulting metrics.
            parameters: Model parameters.
            predictions: List of prediction results.
            training_duration: Training time in seconds.
            data_size: Size of training data.
            horizon: Forecast horizon.
            mlflow_run_id: MLflow run ID if applicable.
            
        Returns:
            Logged ExperimentResult.
        """
        experiment_id = hashlib.md5(
            f"{model_type}_{target_column}_{repository_name}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_type=model_type,
            target_column=target_column,
            repository_name=repository_name,
            metrics=metrics,
            parameters=parameters,
            predictions=predictions,
            training_duration_seconds=training_duration,
            data_size=data_size,
            horizon=horizon,
            mlflow_run_id=mlflow_run_id
        )
        
        self.experiments.append(result)
        self._save_experiments()
        
        logger.info(f"Logged experiment {experiment_id}")
        
        return result
    
    def get_experiments(
        self,
        model_type: Optional[str] = None,
        repository_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ExperimentResult]:
        """Get experiments with optional filtering."""
        results = self.experiments
        
        if model_type:
            results = [e for e in results if e.model_type == model_type]
        
        if repository_name:
            results = [e for e in results if e.repository_name == repository_name]
        
        # Sort by creation time (newest first)
        results = sorted(results, key=lambda e: e.created_at, reverse=True)
        
        return results[:limit]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def get_model_history(self, model_name: str) -> List[ModelVersion]:
        """Get version history for a model."""
        if model_name not in self.models:
            return []
        return sorted(self.models[model_name], key=lambda v: v.created_at)


class MLflowModelRegistry:
    """
    Model registry using MLflow backend.
    Provides more robust tracking and artifact storage.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow model registry.
        
        Args:
            tracking_uri: MLflow tracking URI.
        """
        import mlflow
        
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self._client = mlflow.tracking.MlflowClient()
        logger.info(f"MLflow registry initialized at {self.tracking_uri}")
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> Optional[str]:
        """
        Register a model from an MLflow run.
        
        Args:
            run_id: MLflow run ID.
            model_name: Name for the registered model.
            artifact_path: Path to model artifact in the run.
            
        Returns:
            Model version number if successful.
        """
        import mlflow
        
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered {model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model.
            version: Version to transition.
            stage: New stage (Staging, Production, Archived).
            archive_existing: Whether to archive existing models in target stage.
            
        Returns:
            True if successful.
        """
        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error transitioning stage: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Get the production model."""
        try:
            versions = self._client.get_latest_versions(
                model_name, stages=["Production"]
            )
            if versions:
                return versions[0]
            return None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a model from the registry."""
        import mlflow.pyfunc
        
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def compare_models(
        self,
        experiment_name: str,
        metric: str = "smape"
    ) -> pd.DataFrame:
        """Compare models from an experiment."""
        import mlflow
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"]
        )
        
        return runs


def create_registry(
    use_mlflow: bool = False,
    registry_path: str = "model_registry"
) -> LocalModelRegistry:
    """
    Factory function to create a model registry.
    
    Args:
        use_mlflow: Whether to use MLflow backend.
        registry_path: Path for local registry.
        
    Returns:
        Model registry instance.
    """
    if use_mlflow:
        try:
            return MLflowModelRegistry()
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Using local registry.")
    
    return LocalModelRegistry(registry_path)


if __name__ == "__main__":
    print("Testing Model Registry")
    print("=" * 60)
    
    # Create registry
    registry = LocalModelRegistry("test_registry")
    
    # Register a sample model
    metrics = ModelMetrics(rmse=10.5, mae=8.2, smape=15.3)
    
    model = registry.register_model(
        model_type="prophet",
        target_column="commits",
        repository_name="test_repo",
        metrics=metrics,
        parameters={"seasonality_mode": "additive"},
        description="Test model"
    )
    
    print(f"Registered: {model.version}")
    
    # Register another version
    metrics2 = ModelMetrics(rmse=9.8, mae=7.5, smape=14.1)
    model2 = registry.register_model(
        model_type="prophet",
        target_column="commits",
        repository_name="test_repo",
        metrics=metrics2,
        parameters={"seasonality_mode": "multiplicative"}
    )
    
    print(f"Registered: {model2.version}")
    
    # Promote to production
    model_name = registry._generate_model_name("prophet", "commits", "test_repo")
    registry.promote_to_production(model_name, model2.version)
    
    # Get production model
    prod_model = registry.get_production_model(model_name)
    print(f"Production model: v{prod_model.version}")
    
    # Compare models
    comparison = registry.compare_models([model_name])
    print("\nModel Comparison:")
    print(comparison)
    
    # Cleanup test
    import shutil
    shutil.rmtree("test_registry")
