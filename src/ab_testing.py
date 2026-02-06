"""
A/B Testing Framework - GitHub Activity Predictor
==================================================

This module implements a framework for comparing model versions
in production to evaluate performance before full deployment.

Features:
- Traffic splitting between model versions
- Statistical significance testing
- Experiment tracking and analysis
- Automatic rollback capabilities
- Performance monitoring
"""

import os
import json
import time
import logging
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class TrafficSplitStrategy(Enum):
    """Strategy for splitting traffic between variants."""
    RANDOM = "random"
    STICKY = "sticky"  # Same user always sees same variant
    ROUND_ROBIN = "round_robin"
    PERCENTAGE = "percentage"


@dataclass
class Variant:
    """Represents a model variant in an A/B test."""
    
    name: str
    model_name: str
    model_version: str
    model_type: str
    traffic_percentage: float  # 0.0 to 1.0
    parameters: Dict = field(default_factory=dict)
    is_control: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result metrics from an experiment variant."""
    
    variant_name: str
    n_predictions: int
    total_error: float
    errors: List[float]
    latencies: List[float]  # Prediction latencies in ms
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def mean_error(self) -> float:
        if self.n_predictions == 0:
            return 0.0
        return self.total_error / self.n_predictions
    
    @property
    def mean_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)
    
    @property
    def std_error(self) -> float:
        if len(self.errors) < 2:
            return 0.0
        return np.std(self.errors)
    
    def to_dict(self) -> Dict:
        return {
            "variant_name": self.variant_name,
            "n_predictions": self.n_predictions,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "mean_latency": self.mean_latency,
            "timestamp": self.timestamp
        }


@dataclass
class ABExperiment:
    """Represents an A/B testing experiment."""
    
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    target_column: str
    repository: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    min_samples: int = 100
    confidence_level: float = 0.95
    max_duration_hours: int = 168  # 1 week
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["variants"] = [v.to_dict() for v in self.variants]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ABExperiment":
        data["status"] = ExperimentStatus(data["status"])
        data["variants"] = [Variant(**v) for v in data["variants"]]
        return cls(**data)


@dataclass
class ExperimentAnalysis:
    """Statistical analysis of an A/B experiment."""
    
    experiment_id: str
    control_variant: str
    treatment_variant: str
    control_mean: float
    treatment_mean: float
    relative_improvement: float  # Percentage improvement
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    sample_size_control: int
    sample_size_treatment: int
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TrafficRouter:
    """Routes traffic between variants based on strategy."""
    
    def __init__(self, strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM):
        self.strategy = strategy
        self._round_robin_counter = 0
        self._sticky_assignments: Dict[str, str] = {}
    
    def route(
        self, 
        variants: List[Variant],
        user_id: Optional[str] = None
    ) -> Variant:
        """
        Route a request to a variant.
        
        Args:
            variants: List of available variants.
            user_id: Optional user ID for sticky routing.
            
        Returns:
            Selected variant.
        """
        if not variants:
            raise ValueError("No variants available")
        
        if len(variants) == 1:
            return variants[0]
        
        if self.strategy == TrafficSplitStrategy.STICKY and user_id:
            if user_id in self._sticky_assignments:
                variant_name = self._sticky_assignments[user_id]
                for v in variants:
                    if v.name == variant_name:
                        return v
            
            # Assign new user
            selected = self._select_by_percentage(variants)
            self._sticky_assignments[user_id] = selected.name
            return selected
        
        elif self.strategy == TrafficSplitStrategy.ROUND_ROBIN:
            selected = variants[self._round_robin_counter % len(variants)]
            self._round_robin_counter += 1
            return selected
        
        else:  # RANDOM or PERCENTAGE
            return self._select_by_percentage(variants)
    
    def _select_by_percentage(self, variants: List[Variant]) -> Variant:
        """Select variant based on traffic percentages."""
        rand_val = random.random()
        cumulative = 0.0
        
        for variant in variants:
            cumulative += variant.traffic_percentage
            if rand_val < cumulative:
                return variant
        
        return variants[-1]


class ExperimentStore:
    """Persistent storage for experiments and results."""
    
    def __init__(self, store_path: str = "ab_experiments"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.store_path / "experiments.json"
        self.results_file = self.store_path / "results.json"
        
        self.experiments = self._load_experiments()
        self.results = self._load_results()
    
    def _load_experiments(self) -> Dict[str, ABExperiment]:
        """Load experiments from file."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    data = json.load(f)
                    return {
                        exp_id: ABExperiment.from_dict(exp)
                        for exp_id, exp in data.items()
                    }
            except Exception as e:
                logger.warning(f"Error loading experiments: {e}")
        return {}
    
    def _load_results(self) -> Dict[str, Dict[str, ExperimentResult]]:
        """Load results from file."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r") as f:
                    data = json.load(f)
                    results = {}
                    for exp_id, variants in data.items():
                        results[exp_id] = {
                            v_name: ExperimentResult(
                                variant_name=v_name,
                                n_predictions=v_data["n_predictions"],
                                total_error=v_data.get("total_error", 0),
                                errors=v_data.get("errors", []),
                                latencies=v_data.get("latencies", [])
                            )
                            for v_name, v_data in variants.items()
                        }
                    return results
            except Exception as e:
                logger.warning(f"Error loading results: {e}")
        return {}
    
    def _save_experiments(self):
        """Save experiments to file."""
        data = {exp_id: exp.to_dict() for exp_id, exp in self.experiments.items()}
        with open(self.experiments_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _save_results(self):
        """Save results to file."""
        data = {}
        for exp_id, variants in self.results.items():
            data[exp_id] = {
                v_name: {
                    "n_predictions": r.n_predictions,
                    "total_error": r.total_error,
                    "errors": r.errors[-1000:],  # Keep last 1000
                    "latencies": r.latencies[-1000:]
                }
                for v_name, r in variants.items()
            }
        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def save_experiment(self, experiment: ABExperiment):
        """Save an experiment."""
        self.experiments[experiment.experiment_id] = experiment
        self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        error: float,
        latency: float
    ):
        """Record a prediction result."""
        if experiment_id not in self.results:
            self.results[experiment_id] = {}
        
        if variant_name not in self.results[experiment_id]:
            self.results[experiment_id][variant_name] = ExperimentResult(
                variant_name=variant_name,
                n_predictions=0,
                total_error=0.0,
                errors=[],
                latencies=[]
            )
        
        result = self.results[experiment_id][variant_name]
        result.n_predictions += 1
        result.total_error += error
        result.errors.append(error)
        result.latencies.append(latency)
        
        # Save periodically
        if result.n_predictions % 100 == 0:
            self._save_results()
    
    def get_results(self, experiment_id: str) -> Dict[str, ExperimentResult]:
        """Get results for an experiment."""
        return self.results.get(experiment_id, {})


class ABTestingFramework:
    """
    Main A/B testing framework for comparing model versions.
    """
    
    def __init__(
        self,
        store_path: str = "ab_experiments",
        model_registry_path: str = "model_registry"
    ):
        """
        Initialize the A/B testing framework.
        
        Args:
            store_path: Path for experiment storage.
            model_registry_path: Path to model registry.
        """
        self.store = ExperimentStore(store_path)
        self.router = TrafficRouter(TrafficSplitStrategy.STICKY)
        self.model_registry_path = model_registry_path
        
        self._active_experiments: Dict[str, ABExperiment] = {}
        self._load_active_experiments()
        
        logger.info("A/B Testing Framework initialized")
    
    def _load_active_experiments(self):
        """Load running experiments."""
        for exp_id, exp in self.store.experiments.items():
            if exp.status == ExperimentStatus.RUNNING:
                self._active_experiments[exp_id] = exp
    
    def create_experiment(
        self,
        name: str,
        description: str,
        repository: str,
        target_column: str,
        control_model: Dict[str, str],  # {"name": ..., "version": ..., "type": ...}
        treatment_model: Dict[str, str],
        traffic_split: float = 0.5,  # Percentage for treatment
        min_samples: int = 100,
        confidence_level: float = 0.95
    ) -> ABExperiment:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name.
            description: Experiment description.
            repository: Target repository.
            target_column: Target column for predictions.
            control_model: Control model info.
            treatment_model: Treatment model info.
            traffic_split: Percentage of traffic for treatment (0-1).
            min_samples: Minimum samples before analysis.
            confidence_level: Confidence level for significance.
            
        Returns:
            Created ABExperiment.
        """
        experiment_id = hashlib.md5(
            f"{name}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        variants = [
            Variant(
                name="control",
                model_name=control_model["name"],
                model_version=control_model["version"],
                model_type=control_model["type"],
                traffic_percentage=1.0 - traffic_split,
                is_control=True
            ),
            Variant(
                name="treatment",
                model_name=treatment_model["name"],
                model_version=treatment_model["version"],
                model_type=treatment_model["type"],
                traffic_percentage=traffic_split,
                is_control=False
            )
        ]
        
        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            target_column=target_column,
            repository=repository,
            min_samples=min_samples,
            confidence_level=confidence_level
        )
        
        self.store.save_experiment(experiment)
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        experiment = self.store.get_experiment(experiment_id)
        
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.warning(f"Experiment {experiment_id} is not in draft status")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow().isoformat()
        
        self.store.save_experiment(experiment)
        self._active_experiments[experiment_id] = experiment
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return True
    
    def stop_experiment(
        self, 
        experiment_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED
    ) -> bool:
        """Stop an experiment."""
        experiment = self.store.get_experiment(experiment_id)
        
        if not experiment:
            return False
        
        experiment.status = status
        experiment.end_time = datetime.utcnow().isoformat()
        
        self.store.save_experiment(experiment)
        
        if experiment_id in self._active_experiments:
            del self._active_experiments[experiment_id]
        
        logger.info(f"Stopped experiment: {experiment_id} ({status.value})")
        
        return True
    
    def get_variant_for_request(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Variant]:
        """
        Get the variant for a prediction request.
        
        Args:
            experiment_id: Experiment ID.
            user_id: Optional user ID for sticky routing.
            
        Returns:
            Selected variant or None if experiment not running.
        """
        experiment = self._active_experiments.get(experiment_id)
        
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None
        
        return self.router.route(experiment.variants, user_id)
    
    def record_prediction(
        self,
        experiment_id: str,
        variant_name: str,
        predicted_value: float,
        actual_value: float,
        latency_ms: float
    ):
        """
        Record a prediction result for the experiment.
        
        Args:
            experiment_id: Experiment ID.
            variant_name: Name of variant used.
            predicted_value: Model prediction.
            actual_value: Actual value (ground truth).
            latency_ms: Prediction latency in milliseconds.
        """
        error = abs(predicted_value - actual_value)
        
        self.store.record_result(
            experiment_id=experiment_id,
            variant_name=variant_name,
            error=error,
            latency=latency_ms
        )
    
    def analyze_experiment(
        self,
        experiment_id: str
    ) -> Optional[ExperimentAnalysis]:
        """
        Perform statistical analysis on experiment results.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            ExperimentAnalysis or None if insufficient data.
        """
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return None
        
        results = self.store.get_results(experiment_id)
        
        # Find control and treatment
        control_result = None
        treatment_result = None
        
        for variant in experiment.variants:
            if variant.name in results:
                if variant.is_control:
                    control_result = results[variant.name]
                else:
                    treatment_result = results[variant.name]
        
        if not control_result or not treatment_result:
            logger.warning("Insufficient data for analysis")
            return None
        
        if (control_result.n_predictions < experiment.min_samples or
            treatment_result.n_predictions < experiment.min_samples):
            logger.warning(
                f"Need more samples: control={control_result.n_predictions}, "
                f"treatment={treatment_result.n_predictions}, min={experiment.min_samples}"
            )
            return None
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            control_result.errors,
            treatment_result.errors
        )
        
        is_significant = p_value < (1 - experiment.confidence_level)
        
        # Calculate improvement
        control_mean = control_result.mean_error
        treatment_mean = treatment_result.mean_error
        
        if control_mean > 0:
            relative_improvement = ((control_mean - treatment_mean) / control_mean) * 100
        else:
            relative_improvement = 0.0
        
        # Confidence interval
        pooled_std = np.sqrt(
            (control_result.std_error**2 / control_result.n_predictions) +
            (treatment_result.std_error**2 / treatment_result.n_predictions)
        )
        
        z_score = stats.norm.ppf((1 + experiment.confidence_level) / 2)
        diff = control_mean - treatment_mean
        ci_lower = diff - z_score * pooled_std
        ci_upper = diff + z_score * pooled_std
        
        # Generate recommendation
        if is_significant and treatment_mean < control_mean:
            recommendation = (
                f"PROMOTE treatment model: {relative_improvement:.1f}% improvement "
                f"(p={p_value:.4f})"
            )
        elif is_significant and treatment_mean > control_mean:
            recommendation = (
                f"KEEP control model: Treatment is {-relative_improvement:.1f}% worse "
                f"(p={p_value:.4f})"
            )
        else:
            recommendation = (
                f"NO ACTION: Results not statistically significant (p={p_value:.4f}). "
                f"Continue experiment."
            )
        
        analysis = ExperimentAnalysis(
            experiment_id=experiment_id,
            control_variant="control",
            treatment_variant="treatment",
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            relative_improvement=relative_improvement,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_control=control_result.n_predictions,
            sample_size_treatment=treatment_result.n_predictions,
            recommendation=recommendation
        )
        
        logger.info(f"Analysis for {experiment_id}: {recommendation}")
        
        return analysis
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """Get a summary of the experiment status and results."""
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        results = self.store.get_results(experiment_id)
        
        variant_summaries = []
        for variant in experiment.variants:
            if variant.name in results:
                result = results[variant.name]
                variant_summaries.append({
                    "name": variant.name,
                    "model": f"{variant.model_name} v{variant.model_version}",
                    "is_control": variant.is_control,
                    "traffic_percentage": variant.traffic_percentage,
                    "n_predictions": result.n_predictions,
                    "mean_error": result.mean_error,
                    "mean_latency_ms": result.mean_latency
                })
        
        # Try to get analysis
        analysis = None
        try:
            analysis = self.analyze_experiment(experiment_id)
        except Exception:
            pass
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "repository": experiment.repository,
            "target_column": experiment.target_column,
            "start_time": experiment.start_time,
            "end_time": experiment.end_time,
            "variants": variant_summaries,
            "analysis": analysis.to_dict() if analysis else None
        }
    
    def auto_rollback(
        self,
        experiment_id: str,
        error_threshold: float,
        sample_window: int = 50
    ) -> bool:
        """
        Check if treatment should be rolled back due to poor performance.
        
        Args:
            experiment_id: Experiment ID.
            error_threshold: Maximum acceptable error.
            sample_window: Number of recent samples to check.
            
        Returns:
            True if rollback was triggered.
        """
        results = self.store.get_results(experiment_id)
        
        treatment_result = results.get("treatment")
        if not treatment_result or len(treatment_result.errors) < sample_window:
            return False
        
        recent_errors = treatment_result.errors[-sample_window:]
        recent_mean = np.mean(recent_errors)
        
        if recent_mean > error_threshold:
            logger.warning(
                f"Auto-rollback triggered for {experiment_id}: "
                f"recent error {recent_mean:.2f} > threshold {error_threshold}"
            )
            
            self.stop_experiment(experiment_id, ExperimentStatus.ABORTED)
            return True
        
        return False
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[ABExperiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self.store.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)


def create_ab_framework(
    store_path: str = "ab_experiments"
) -> ABTestingFramework:
    """Factory function to create A/B testing framework."""
    return ABTestingFramework(store_path=store_path)


if __name__ == "__main__":
    print("Testing A/B Testing Framework")
    print("=" * 60)
    
    # Create framework
    framework = create_ab_framework("test_ab_experiments")
    
    # Create experiment
    experiment = framework.create_experiment(
        name="Prophet vs FARIMA",
        description="Compare Prophet and FARIMA models for commit prediction",
        repository="test_repo",
        target_column="commits",
        control_model={
            "name": "test_repo_commits_prophet",
            "version": "1.0.0",
            "type": "prophet"
        },
        treatment_model={
            "name": "test_repo_commits_farima",
            "version": "1.0.0",
            "type": "farima"
        },
        traffic_split=0.5
    )
    
    print(f"Created experiment: {experiment.experiment_id}")
    
    # Start experiment
    framework.start_experiment(experiment.experiment_id)
    
    # Simulate predictions
    import random
    
    for i in range(200):
        variant = framework.get_variant_for_request(
            experiment.experiment_id,
            user_id=f"user_{i % 20}"
        )
        
        if variant:
            # Simulate prediction
            if variant.is_control:
                error = random.gauss(10, 3)  # Control: mean error 10
            else:
                error = random.gauss(8, 3)  # Treatment: mean error 8 (better)
            
            predicted = 100
            actual = 100 + error
            
            framework.record_prediction(
                experiment_id=experiment.experiment_id,
                variant_name=variant.name,
                predicted_value=predicted,
                actual_value=actual,
                latency_ms=random.gauss(50, 10)
            )
    
    # Analyze
    analysis = framework.analyze_experiment(experiment.experiment_id)
    
    print("\nAnalysis Results:")
    print(f"  Control mean error: {analysis.control_mean:.2f}")
    print(f"  Treatment mean error: {analysis.treatment_mean:.2f}")
    print(f"  Improvement: {analysis.relative_improvement:.1f}%")
    print(f"  P-value: {analysis.p_value:.4f}")
    print(f"  Significant: {analysis.is_significant}")
    print(f"  Recommendation: {analysis.recommendation}")
    
    # Get summary
    summary = framework.get_experiment_summary(experiment.experiment_id)
    print("\nExperiment Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Stop experiment
    framework.stop_experiment(experiment.experiment_id)
    
    # Cleanup
    import shutil
    shutil.rmtree("test_ab_experiments")
