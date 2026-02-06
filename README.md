# GitHub Activity Predictor v2.0

**Enterprise-Grade MLOps Pipeline for GitHub Repository Activity Forecasting**

A complete MLOps solution that predicts GitHub repository activity (commits, stars, issues, PRs) using auto-regressive forecasting with **Prophet**, **FARIMA**, and **Neural Network** models.

## New in v2.0

- **Neural Network Model** - Exploits correlation between PRs and commits
- **Auto Model Selection** - Automatic hyperparameter optimization
- **Confidence Intervals** - Quantify prediction uncertainty (95% CI)
- **Data Validation** - Schema validation, anomaly detection, drift monitoring
- **Model Registry** - Version control and lifecycle management
- **REST API** - FastAPI server with authentication and rate limiting
- **Workflow Orchestration** - Prefect-based pipeline scheduling
- **A/B Testing Framework** - Compare model versions in production
- **Distributed Computing** - Dask support for scalable processing
- **CI/CD Pipeline** - GitHub Actions for automated testing and deployment

---

## Quick Start

### 1. Install Dependencies
```bash
cd /info/raid-etu/m2/s2405959/BigData/last_update/bigmlops
pip install -e .

# Optional: Install neural network support
pip install -e ".[neural]"

# Optional: Install distributed computing support
pip install -e ".[distributed]"

# Install all optional dependencies
pip install -e ".[all]"
```

### 2. Run Services

**Option A: Docker Compose (Recommended)**
```bash
docker compose up --build
```

**Option B: Local Development**

```bash
# Terminal 1 - MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Terminal 2 - Streamlit Dashboard
streamlit run src/dashboard.py --server.port 8501

# Terminal 3 - FastAPI Server (optional)
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

---

## Access URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:8501 | Streamlit UI for predictions |
| **MLflow** | http://localhost:5000 | Experiment tracking |
| **API Server** | http://localhost:8000 | REST API for predictions |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Orchestration                         │
├──────────────┬──────────────┬──────────────┬───────────────────────────────┤
│   Dashboard  │   API Server │    MLflow    │      Prefect Orchestrator     │
│  (Port 8501) │  (Port 8000) │ (Port 5000)  │        (Port 4200)            │
├──────────────┴──────────────┴──────────────┴───────────────────────────────┤
│                              Core Modules                                    │
├─────────────────┬─────────────────┬─────────────────┬──────────────────────┤
│   ETL Module    │  Model Engine   │  Neural Network │   Model Selection    │
│   (etl.py)      │(model_engine.py)│(neural_network) │ (model_selection.py) │
├─────────────────┴─────────────────┴─────────────────┴──────────────────────┤
│                           Support Modules                                    │
├────────────────┬────────────────┬─────────────────┬────────────────────────┤
│ Data Ingestion │ Data Validation│  Model Registry │     A/B Testing       │
│(data_ingestion)│(data_validation│ (model_registry)│   (ab_testing.py)     │
└────────────────┴────────────────┴─────────────────┴────────────────────────┘
```

## Project Structure

```
bigmlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yaml         # GitHub Actions CI/CD pipeline
├── src/
│   ├── __init__.py
│   ├── etl.py                 # ETL: Extraction & Cleaning
│   ├── model_engine.py        # Prophet & FARIMA models
│   ├── neural_network.py      # Neural network for commits prediction
│   ├── model_selection.py     # Auto model selection & hyperparameter tuning
│   ├── data_validation.py     # Schema, anomaly, drift detection
│   ├── model_registry.py      # Model versioning & lifecycle
│   ├── data_ingestion.py      # GitHub API data fetching
│   ├── api_server.py          # FastAPI prediction server
│   ├── orchestration.py       # Prefect workflow orchestration
│   ├── ab_testing.py          # A/B testing framework
│   ├── distributed.py         # Dask distributed computing
│   └── dashboard.py           # Streamlit UI
├── repositories/              # Repository data (CSV files)
├── mlruns/                    # MLflow tracking data
├── model_registry/            # Registered model versions
├── ab_experiments/            # A/B test data
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```
├── pyproject.toml           # uv/pip dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+ (for local development)
- `uv` package manager (optional, for local dev)

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to project
cd github-activity-predictor

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Access the applications:
# - Streamlit Dashboard: http://localhost:8501
# - MLflow UI: http://localhost:5000
```

### Local Development

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

uv pip install -e .

# Start MLflow server (in a separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# Run Streamlit dashboard
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

- **Automatic Hyperparameter Tuning**: Grid search with cross-validation
- **Data Characteristic Analysis**: Recommends models based on data properties
- **Multi-metric Evaluation**: RMSE, MAE, SMAPE comparison
- **Best Model Selection**: Automatic selection of optimal model

### Data Validation (`src/data_validation.py`)

- **Schema Validation**: Column types, required fields, value ranges
- **Anomaly Detection**: IQR and Z-score methods
- **Data Drift Monitoring**: KS-test and PSI metrics
- **Quality Reports**: Comprehensive validation summaries

### Model Registry (`src/model_registry.py`)

- **Semantic Versioning**: Major.minor.patch version control
- **Lifecycle Stages**: Staging → Production → Archived
- **MLflow Integration**: Optional cloud-based registry
- **Model Metadata**: Parameters, metrics, tags

### API Server (`src/api_server.py`)

- **REST API**: FastAPI with automatic OpenAPI docs
- **Authentication**: API key-based auth
- **Rate Limiting**: Configurable request limits
- **Batch Predictions**: Multiple repository predictions
- **Health Checks**: Service status endpoints

### Orchestration (`src/orchestration.py`)

- **Prefect Workflows**: Scheduled data ingestion and training
- **Task Retries**: Automatic retry on failures
- **Notifications**: Email/Slack on completion
- **Pipeline Monitoring**: Execution tracking

### A/B Testing (`src/ab_testing.py`)

- **Traffic Splitting**: Random, sticky, or percentage-based
- **Statistical Analysis**: T-test significance testing
- **Auto-rollback**: Performance-based rollback
- **Experiment Tracking**: Persistent result storage

### Distributed Computing (`src/distributed.py`)

- **Dask Integration**: Parallel data processing
- **Distributed Training**: Multi-worker model training
- **Memory Efficiency**: Handle large datasets
- **Cluster Management**: Easy worker scaling

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

## CI/CD Pipeline

The project includes GitHub Actions workflows for:

1. **Linting**: ruff, black, isort, mypy
2. **Testing**: pytest with coverage
3. **Data Validation**: Quality checks on repository data
4. **Model Training**: Scheduled weekly training
5. **Docker Build**: Container image creation
6. **Deployment**: Staging and production deployments

### Secrets Required

| Secret | Description |
|--------|-------------|
| `GH_API_TOKEN` | GitHub API token for data ingestion |
| `MLFLOW_TRACKING_URI` | MLflow server URL |

---

## Recursive Forecasting Algorithm

The core algorithm implements "blind forecasting" - simulating what would happen if no new data were available:

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

This approach is more realistic because:
- Error compounds over time (as in real scenarios)
- Model adapts based on its own predictions
- Reflects true uncertainty in longer forecasts

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `GITHUB_TOKEN` | - | GitHub API token for data ingestion |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |

### Model Parameters

#### Prophet
- `yearly_seasonality`: True
- `weekly_seasonality`: True  
- `seasonality_mode`: multiplicative

#### FARIMA
- `order`: (1, 1, 1)
- `seasonal_order`: (1, 1, 1, 52)

#### Neural Network
- `hidden_layers`: [64, 32, 16]
- `activation`: ReLU
- `epochs`: 200
- `learning_rate`: 0.001

## Data Format

### Input CSV Files

Each repository folder should contain:

| File | Required Column | Description |
|------|-----------------|-------------|
| `commits.csv` | `author_date` | Commit timestamps |
| `issues.csv` | `created_at` | Issue creation dates |
| `pull_requests.csv` | `created_at` | PR creation dates |
| `stargazers.csv` | `starred_at` | Star timestamps |

### Output DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | Week end date (Sunday) |
| `commits` | int | Number of commits that week |
| `new_stars` | int | New stars received |
| `issues_opened` | int | Issues opened |
| `prs_opened` | int | Pull requests opened |

## Docker Commands

```bash
# Build and start all services
docker compose up --build -d

# Start specific services
docker compose up dashboard mlflow -d

# View logs
docker compose logs -f dashboard
docker compose logs -f api

# Stop services
docker compose down

# Clean up everything
docker compose down -v --rmi all
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request

---

Built with Python, Streamlit, Prophet, FARIMA, PyTorch, FastAPI, Prefect, MLflow & Docker
