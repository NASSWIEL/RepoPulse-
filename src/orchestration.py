"""
Pipeline Orchestration Module - GitHub Activity Predictor
==========================================================

This module implements workflow orchestration using Prefect to schedule
and monitor data ingestion and model training jobs.

Features:
- Scheduled data ingestion pipelines
- Automated model training workflows
- Task dependencies and error handling
- Monitoring and alerting
- Retry logic for resilience
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Ingestion Tasks
# =============================================================================

@task(
    name="fetch_repository_data",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24)
)
def fetch_repository_data(
    owner: str,
    repo: str,
    github_token: str,
    data_dir: str = "repositories"
) -> Dict:
    """
    Fetch data for a single repository from GitHub.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        github_token: GitHub API token.
        data_dir: Directory to store data.
        
    Returns:
        Fetch result dictionary.
    """
    log = get_run_logger()
    log.info(f"Fetching repository: {owner}/{repo}")
    
    from src.data_ingestion import DataIngestionPipeline
    
    pipeline = DataIngestionPipeline(
        github_token=github_token,
        data_dir=data_dir
    )
    
    result = pipeline.fetch_repository(owner, repo, incremental=True)
    
    if result["status"] == "success":
        log.info(f"Successfully fetched {owner}/{repo}: {result.get('record_counts', {})}")
    elif result["status"] == "skipped":
        log.info(f"Skipped {owner}/{repo}: {result.get('reason', 'unknown')}")
    else:
        log.error(f"Failed to fetch {owner}/{repo}: {result.get('error', 'unknown')}")
    
    return result


@task(
    name="validate_repository_data",
    retries=2,
    retry_delay_seconds=30
)
def validate_repository_data(
    repository: str,
    data_dir: str = "repositories"
) -> Dict:
    """
    Validate data quality for a repository.
    
    Args:
        repository: Repository folder name.
        data_dir: Directory containing repository data.
        
    Returns:
        Validation report dictionary.
    """
    log = get_run_logger()
    log.info(f"Validating data for: {repository}")
    
    from src.etl import load_repository_data
    from src.data_validation import create_validator
    
    repo_path = Path(data_dir) / repository
    
    try:
        df = load_repository_data(str(repo_path))
        
        validator = create_validator()
        report = validator.validate(df, repository)
        
        log.info(f"Validation score: {report.overall_score:.1f}/100")
        
        return report.to_dict()
        
    except Exception as e:
        log.error(f"Validation failed: {e}")
        return {"error": str(e), "passed": False}


@task(
    name="load_repository_list",
    retries=2
)
def load_repository_list(
    summary_file: str = "repositories_summary.csv",
    limit: Optional[int] = None
) -> List[tuple]:
    """
    Load list of repositories to process.
    
    Args:
        summary_file: Path to summary CSV file.
        limit: Maximum repositories to return.
        
    Returns:
        List of (owner, repo) tuples.
    """
    log = get_run_logger()
    
    df = pd.read_csv(summary_file)
    
    repositories = []
    for _, row in df.iterrows():
        full_name = row.get("full_name", "")
        if "/" in full_name:
            owner, repo = full_name.split("/", 1)
            repositories.append((owner, repo))
    
    if limit:
        repositories = repositories[:limit]
    
    log.info(f"Loaded {len(repositories)} repositories")
    
    return repositories


# =============================================================================
# Model Training Tasks
# =============================================================================

@task(
    name="train_model",
    retries=2,
    retry_delay_seconds=120,
    timeout_seconds=600
)
def train_model(
    repository: str,
    target_column: str = "commits",
    model_type: Optional[str] = None,
    horizon: int = 8,
    data_dir: str = "repositories"
) -> Dict:
    """
    Train a forecasting model for a repository.
    
    Args:
        repository: Repository folder name.
        target_column: Column to predict.
        model_type: Model type or None for auto-selection.
        horizon: Forecast horizon.
        data_dir: Directory containing repository data.
        
    Returns:
        Training result dictionary.
    """
    log = get_run_logger()
    log.info(f"Training model for {repository}, target={target_column}")
    
    from src.etl import load_repository_data
    
    repo_path = Path(data_dir) / repository
    df = load_repository_data(str(repo_path))
    
    if model_type is None:
        # Auto model selection
        from src.model_selection import auto_select_model
        
        result = auto_select_model(
            df=df,
            target_column=target_column,
            horizon=horizon,
            quick_mode=True
        )
        
        log.info(f"Best model: {result.best_model_type}, SMAPE: {result.best_metrics['smape']:.2f}%")
        
        return {
            "repository": repository,
            "model_type": result.best_model_type,
            "parameters": result.best_parameters,
            "metrics": result.best_metrics,
            "selection_reason": result.selection_reason
        }
    else:
        # Specific model
        from src.model_engine import train_predict_recursive
        
        result = train_predict_recursive(
            df=df,
            target_column=target_column,
            model_type=model_type,
            horizon=horizon,
            repo_name=repository
        )
        
        log.info(f"Model trained: SMAPE={result['metrics']['smape']:.2f}%")
        
        return {
            "repository": repository,
            "model_type": model_type,
            "metrics": result["metrics"],
            "mlflow_run_id": result.get("mlflow_run_id")
        }


@task(
    name="register_model",
    retries=2
)
def register_model(
    training_result: Dict,
    registry_path: str = "model_registry"
) -> Dict:
    """
    Register a trained model in the registry.
    
    Args:
        training_result: Result from train_model task.
        registry_path: Path to model registry.
        
    Returns:
        Registration result.
    """
    log = get_run_logger()
    
    from src.model_registry import LocalModelRegistry, ModelMetrics
    
    registry = LocalModelRegistry(registry_path)
    
    metrics = ModelMetrics(
        rmse=training_result["metrics"]["rmse"],
        mae=training_result["metrics"]["mae"],
        smape=training_result["metrics"]["smape"]
    )
    
    model = registry.register_model(
        model_type=training_result["model_type"],
        target_column=training_result.get("target_column", "commits"),
        repository_name=training_result["repository"],
        metrics=metrics,
        parameters=training_result.get("parameters", {}),
        mlflow_run_id=training_result.get("mlflow_run_id")
    )
    
    log.info(f"Registered model version {model.version}")
    
    return {
        "model_name": registry._generate_model_name(
            model.model_type, 
            model.target_column, 
            model.repository_name
        ),
        "version": model.version,
        "stage": model.stage.value
    }


@task(
    name="compare_models",
    retries=1
)
def compare_models(
    repository: str,
    target_column: str = "commits",
    registry_path: str = "model_registry"
) -> Dict:
    """
    Compare model versions and determine best.
    
    Args:
        repository: Repository name.
        target_column: Target column.
        registry_path: Path to model registry.
        
    Returns:
        Comparison results.
    """
    log = get_run_logger()
    
    from src.model_registry import LocalModelRegistry
    
    registry = LocalModelRegistry(registry_path)
    
    # Get all models for this repo/target
    model_names = [
        f"{repository}_{target_column}_{mt}"
        for mt in ["prophet", "farima", "neural_network"]
    ]
    
    comparison = registry.compare_models(model_names)
    
    if len(comparison) > 0:
        best = comparison.iloc[0]
        log.info(f"Best model: {best['model_name']} v{best['version']} (SMAPE={best['smape']:.2f}%)")
        
        return {
            "best_model": best["model_name"],
            "best_version": best["version"],
            "best_smape": best["smape"],
            "comparison": comparison.to_dict("records")
        }
    
    return {"error": "No models found for comparison"}


@task(
    name="send_notification",
    retries=2
)
def send_notification(
    message: str,
    level: str = "info",
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a notification (e.g., Slack, email).
    
    Args:
        message: Notification message.
        level: Message level (info, warning, error).
        webhook_url: Webhook URL for notification.
        
    Returns:
        Success status.
    """
    log = get_run_logger()
    log.info(f"[{level.upper()}] {message}")
    
    if webhook_url:
        import requests
        
        try:
            requests.post(webhook_url, json={
                "text": f"[{level.upper()}] {message}",
                "timestamp": datetime.utcnow().isoformat()
            }, timeout=10)
            return True
        except Exception as e:
            log.warning(f"Failed to send webhook: {e}")
    
    return True


# =============================================================================
# Flows
# =============================================================================

@flow(
    name="data_ingestion_flow",
    description="Fetch and validate data for repositories"
)
def data_ingestion_flow(
    repositories: Optional[List[tuple]] = None,
    github_token: Optional[str] = None,
    data_dir: str = "repositories",
    summary_file: str = "repositories_summary.csv",
    limit: Optional[int] = None
) -> Dict:
    """
    Main data ingestion flow.
    
    Args:
        repositories: List of (owner, repo) tuples. If None, loads from summary.
        github_token: GitHub API token.
        data_dir: Data directory.
        summary_file: Path to summary file.
        limit: Maximum repositories to process.
        
    Returns:
        Flow results.
    """
    log = get_run_logger()
    log.info("Starting data ingestion flow")
    
    # Get token from env if not provided
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token required")
    
    # Load repositories if not provided
    if repositories is None:
        repositories = load_repository_list(summary_file, limit)
    
    # Fetch each repository
    results = []
    for owner, repo in repositories:
        result = fetch_repository_data(owner, repo, token, data_dir)
        results.append(result)
    
    # Validate fetched data
    validation_results = []
    for result in results:
        if result["status"] == "success":
            folder = Path(result["folder"]).name
            validation = validate_repository_data(folder, data_dir)
            validation_results.append(validation)
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "error")
    
    summary = {
        "total_repositories": len(repositories),
        "success": success_count,
        "failed": failed_count,
        "skipped": len(results) - success_count - failed_count,
        "validation_results": validation_results
    }
    
    log.info(f"Ingestion complete: {success_count} success, {failed_count} failed")
    
    # Send notification if failures
    if failed_count > 0:
        send_notification(
            f"Data ingestion completed with {failed_count} failures",
            level="warning"
        )
    
    return summary


@flow(
    name="model_training_flow",
    description="Train and register models for repositories"
)
def model_training_flow(
    repositories: List[str],
    target_columns: List[str] = None,
    model_type: Optional[str] = None,
    horizon: int = 8,
    data_dir: str = "repositories",
    registry_path: str = "model_registry"
) -> Dict:
    """
    Model training flow.
    
    Args:
        repositories: List of repository folder names.
        target_columns: List of columns to predict.
        model_type: Model type or None for auto-selection.
        horizon: Forecast horizon.
        data_dir: Data directory.
        registry_path: Registry path.
        
    Returns:
        Flow results.
    """
    log = get_run_logger()
    log.info(f"Starting model training flow for {len(repositories)} repositories")
    
    target_columns = target_columns or ["commits"]
    
    all_results = []
    
    for repo in repositories:
        for target in target_columns:
            try:
                # Train model
                training_result = train_model(
                    repository=repo,
                    target_column=target,
                    model_type=model_type,
                    horizon=horizon,
                    data_dir=data_dir
                )
                training_result["target_column"] = target
                
                # Register model
                registration = register_model(training_result, registry_path)
                
                all_results.append({
                    "repository": repo,
                    "target": target,
                    "status": "success",
                    "training": training_result,
                    "registration": registration
                })
                
            except Exception as e:
                log.error(f"Failed to train model for {repo}/{target}: {e}")
                all_results.append({
                    "repository": repo,
                    "target": target,
                    "status": "error",
                    "error": str(e)
                })
    
    # Summary
    success = sum(1 for r in all_results if r["status"] == "success")
    
    summary = {
        "total_models": len(all_results),
        "success": success,
        "failed": len(all_results) - success,
        "results": all_results
    }
    
    log.info(f"Training complete: {success}/{len(all_results)} models trained")
    
    return summary


@flow(
    name="full_pipeline_flow",
    description="Complete pipeline: ingestion, validation, training, registration"
)
def full_pipeline_flow(
    github_token: Optional[str] = None,
    data_dir: str = "repositories",
    summary_file: str = "repositories_summary.csv",
    limit: Optional[int] = 10,
    target_columns: List[str] = None,
    model_type: Optional[str] = None,
    horizon: int = 8
) -> Dict:
    """
    Full pipeline flow combining ingestion and training.
    
    Args:
        github_token: GitHub API token.
        data_dir: Data directory.
        summary_file: Summary file path.
        limit: Max repositories to process.
        target_columns: Columns to predict.
        model_type: Model type.
        horizon: Forecast horizon.
        
    Returns:
        Complete pipeline results.
    """
    log = get_run_logger()
    log.info("Starting full pipeline flow")
    
    # Step 1: Data Ingestion
    ingestion_result = data_ingestion_flow(
        github_token=github_token,
        data_dir=data_dir,
        summary_file=summary_file,
        limit=limit
    )
    
    # Get successfully ingested repositories
    repos_dir = Path(data_dir)
    repositories = [
        d.name for d in repos_dir.iterdir() 
        if d.is_dir() and not d.name.startswith(".")
    ][:limit or 1000]
    
    # Step 2: Model Training
    training_result = model_training_flow(
        repositories=repositories,
        target_columns=target_columns,
        model_type=model_type,
        horizon=horizon,
        data_dir=data_dir
    )
    
    # Step 3: Model Comparison (for each repo)
    comparison_results = []
    for repo in repositories[:5]:  # Limit comparisons
        try:
            comparison = compare_models(
                repository=repo,
                target_column=(target_columns or ["commits"])[0]
            )
            comparison_results.append(comparison)
        except Exception:
            pass
    
    # Final summary
    summary = {
        "ingestion": ingestion_result,
        "training": training_result,
        "comparisons": comparison_results,
        "completed_at": datetime.utcnow().isoformat()
    }
    
    send_notification(
        f"Pipeline completed: {ingestion_result['success']} repos ingested, "
        f"{training_result['success']} models trained",
        level="info"
    )
    
    return summary


# =============================================================================
# Deployment Configuration
# =============================================================================

def create_deployments():
    """Create Prefect deployments for scheduled execution."""
    
    # Daily data ingestion
    ingestion_deployment = Deployment.build_from_flow(
        flow=data_ingestion_flow,
        name="daily-data-ingestion",
        schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
        parameters={
            "limit": 100,
            "data_dir": "repositories"
        },
        tags=["data", "ingestion", "scheduled"]
    )
    
    # Weekly model training
    training_deployment = Deployment.build_from_flow(
        flow=model_training_flow,
        name="weekly-model-training",
        schedule=CronSchedule(cron="0 4 * * 0"),  # 4 AM Sundays
        parameters={
            "target_columns": ["commits"],
            "horizon": 8
        },
        tags=["training", "scheduled"]
    )
    
    # Full pipeline (weekly)
    full_pipeline_deployment = Deployment.build_from_flow(
        flow=full_pipeline_flow,
        name="weekly-full-pipeline",
        schedule=CronSchedule(cron="0 0 * * 0"),  # Midnight Sundays
        parameters={
            "limit": 50,
            "target_columns": ["commits"],
            "horizon": 8
        },
        tags=["pipeline", "full", "scheduled"]
    )
    
    return [ingestion_deployment, training_deployment, full_pipeline_deployment]


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Orchestration")
    parser.add_argument("command", choices=["ingest", "train", "full", "deploy"])
    parser.add_argument("--token", help="GitHub token")
    parser.add_argument("--limit", type=int, default=10, help="Max repositories")
    parser.add_argument("--repos", nargs="+", help="Specific repositories")
    parser.add_argument("--target", default="commits", help="Target column")
    parser.add_argument("--model", help="Model type")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        result = data_ingestion_flow(
            github_token=args.token,
            limit=args.limit
        )
        print(f"Ingestion result: {result}")
    
    elif args.command == "train":
        repos = args.repos or ["3b1b__manim"]
        result = model_training_flow(
            repositories=repos,
            target_columns=[args.target],
            model_type=args.model
        )
        print(f"Training result: {result}")
    
    elif args.command == "full":
        result = full_pipeline_flow(
            github_token=args.token,
            limit=args.limit,
            target_columns=[args.target],
            model_type=args.model
        )
        print(f"Pipeline result: {result}")
    
    elif args.command == "deploy":
        deployments = create_deployments()
        for dep in deployments:
            dep.apply()
        print(f"Created {len(deployments)} deployments")
