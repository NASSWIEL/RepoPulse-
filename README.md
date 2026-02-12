# GitHub Activity Predictor

Forecast GitHub repository activity (commits, stars, issues, pull requests) using time-series and neural network models.

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

# Run API
uvicorn src.api_server:app --port 8000
```

---



---

## Forecasting Method

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
