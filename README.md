# GitHub Activity Predictor

MLOps pipeline for predicting GitHub repository activity (commits, stars, issues, PRs) using auto-regressive forecasting with **Prophet**, **FARIMA**, and **Neural Network** models.

## Demo

![Demo Interface](images/shot_demo.png)

## New in v2.0

- Neural Network Model - Uses correlation between PRs and commits
- Auto Model Selection - Automatic hyperparameter optimization
- Confidence Intervals - Quantify prediction uncertainty (95% CI)
- Data Validation - Schema validation, anomaly detection, drift monitoring
- Model Registry - Version control and lifecycle management
- REST API - FastAPI server with authentication and rate limiting
- Workflow Orchestration - Prefect-based pipeline scheduling
- A/B Testing Framework - Compare model versions in production
- Distributed Computing - Dask support for scalable processing
- CI/CD Pipeline - GitHub Actions for automated testing and deployment

---

## Overview

This project provides a complete pipeline for:

- Data ingestion from GitHub repositories  
- Weekly aggregation and validation  
- Time-series forecasting  
- Model comparison and selection  
- Experiment tracking  
- REST API access  
- Dashboard visualization  
- Containerized deployment  



## Features

- Automatic model selection with cross-validation  
- 95% confidence intervals  
- Data validation (schema checks, anomaly detection, drift monitoring)  
- Model versioning and lifecycle management  
- REST API with authentication and rate limiting  
- Workflow orchestration  
- A/B testing support  
- Optional distributed processing  
- CI/CD integration  

---

## Project Structure

```
bigmlops/
├── src/
│   ├── etl.py
│   ├── model_engine.py
│   ├── neural_network.py
│   ├── model_selection.py
│   ├── data_validation.py
│   ├── model_registry.py
│   ├── data_ingestion.py
│   ├── api_server.py
│   ├── orchestration.py
│   ├── ab_testing.py
│   ├── distributed.py
│   └── dashboard.py
├── repositories/
├── mlruns/
├── model_registry/
├── ab_experiments/
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

---

## Installation

### Using Docker (Recommended)

```bash
docker compose up --build -d
```

Access:

- Dashboard: http://localhost:8501  
- MLflow: http://localhost:5000  
- API: http://localhost:8000  

---

### Local Development

```bash
pip install -e .

# Start MLflow
mlflow ui --port 5000

# Run dashboard
streamlit run src/dashboard.py
```

## Features

### Forecasting Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Prophet** | Facebook's decomposition model | Strong seasonality |
| **FARIMA** | Fractional differencing for long memory | Long-range dependencies |
| **Neural Network** | MLP exploiting PRs-commits correlation | Complex patterns |

### Model Selection (`src/model_selection.py`)

- Automatic Hyperparameter Tuning: Grid search with cross-validation
- Data Characteristic Analysis: Recommends models based on data properties
- Multi-metric Evaluation: RMSE, MAE, SMAPE comparison
- Best Model Selection: Automatic selection of optimal model

### Data Validation (`src/data_validation.py`)

- Schema Validation: Column types, required fields, value ranges
- Anomaly Detection: IQR and Z-score methods
- Data Drift Monitoring: KS-test and PSI metrics
- Quality Reports: Comprehensive validation summaries

### Model Registry (`src/model_registry.py`)

- Semantic Versioning: Major.minor.patch version control
- Lifecycle Stages: Staging → Production → Archived
- MLflow Integration: Optional cloud-based registry
- Model Metadata: Parameters, metrics, tags

### API Server (`src/api_server.py`)

- REST API: FastAPI with automatic OpenAPI docs
- Authentication: API key-based auth
- Rate Limiting: Configurable request limits
- Batch Predictions: Multiple repository predictions
- Health Checks: Service status endpoints

### Orchestration (`src/orchestration.py`)

- Prefect Workflows: Scheduled data ingestion and training
- Task Retries: Automatic retry on failures
- Notifications: Email/Slack on completion
- Pipeline Monitoring: Execution tracking

### A/B Testing (`src/ab_testing.py`)

- Traffic Splitting: Random, sticky, or percentage-based
- Statistical Analysis: T-test significance testing
- Auto-rollback: Performance-based rollback
- Experiment Tracking: Persistent result storage

### Distributed Computing (`src/distributed.py`)

- Dask Integration: Parallel data processing
- Distributed Training: Multi-worker model training
- Memory Efficiency: Handle large datasets
- Cluster Management: Easy worker scaling

---

## API Usage

### Authentication

```bash
# Get API key (default: demo-api-key-12345)
export API_KEY="demo-api-key-12345"
```

### Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List repositories
curl -H "X-API-Key: $API_KEY" http://localhost:8000/repositories

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "tensorflow__tensorflow",
    "target_column": "commits",
    "model_type": "prophet",
    "horizon": 8,
    "include_confidence_intervals": true
  }'

# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"repository": "repo1", "target_column": "commits"},
      {"repository": "repo2", "target_column": "commits"}
    ]
  }'
```

---



---

## Recursive Forecasting Algorithm

The algorithm implements "blind forecasting" - simulating what happens when no new data is available:

```
1. Split: Train (historical) / Test (horizon h weeks)

2. For each step t from 1 to h:
   ├── Train model on current history
   ├── Predict only t+1
   ├── CRITICAL: Add ŷ_{t+1} to history (as ground truth)
   └── Repeat

3. Calculate metrics on test period
4. Log everything to MLflow
```

This approach is realistic because:
- Error compounds over time
- Model adapts based on its own predictions
- Reflects uncertainty in longer forecasts

The system uses recursive multi-step forecasting:

1. Train on historical data  
2. Predict the next step  
3. Append prediction to history  
4. Repeat for the full horizon  

This setup reflects real deployment conditions where future ground truth is not available.

---

## Data Requirements

Each repository directory should contain:

- `commits.csv`
- `issues.csv`
- `pull_requests.csv`
- `stargazers.csv`

### Weekly Output Format

| Column          | Description             |
|-----------------|-------------------------|
| `ds`            | Week date               |
| `commits`       | Number of commits       |
| `new_stars`     | New stars received      |
| `issues_opened` | Issues opened           |
| `prs_opened`    | Pull requests opened    |

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

---

## License

MIT License
