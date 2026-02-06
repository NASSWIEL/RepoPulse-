# ================================================
# GitHub Activity Predictor - Dockerfile (v2.0)
# ================================================
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core dependencies with uv
RUN uv pip install --no-cache \
    # Core data processing
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    pyarrow>=15.0.0 \
    # Forecasting models
    prophet>=1.1.5 \
    statsmodels>=0.14.0 \
    scikit-learn>=1.4.0 \
    # Experiment tracking
    mlflow>=2.10.0 \
    # Dashboard
    streamlit>=1.30.0 \
    plotly>=5.18.0 \
    # API serving
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.5.0 \
    # Workflow orchestration
    prefect>=2.14.0 \
    # Data ingestion & utilities
    requests>=2.31.0 \
    scipy>=1.12.0

# ================================================
# Production Stage
# ================================================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create data and storage directories
RUN mkdir -p /app/data/repositories /app/mlruns /app/model_registry /app/ab_experiments

# Expose ports (Streamlit, API, Prefect)
EXPOSE 8501 8000 4200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
