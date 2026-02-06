"""
API Serving Layer - GitHub Activity Predictor
==============================================

This module implements a REST API to serve model predictions
to downstream applications using FastAPI.

Features:
- RESTful API for predictions
- Authentication with API keys
- Rate limiting
- Model registry integration
- Batch prediction support
- Health checks and monitoring
"""

import os
import time
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, Query, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request for a single prediction."""
    
    repository: str = Field(..., description="Repository name (owner__repo format)")
    target_column: str = Field(
        default="commits",
        description="Metric to predict (commits, new_stars, issues_opened, prs_opened)"
    )
    horizon: int = Field(default=4, ge=1, le=52, description="Forecast horizon in weeks")
    model_type: Optional[str] = Field(
        default=None,
        description="Model type (prophet, farima, neural_network). Auto-selected if None."
    )
    include_intervals: bool = Field(
        default=True,
        description="Include prediction intervals"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    requests: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """Response for a prediction request."""
    
    repository: str
    target_column: str
    model_type: str
    predictions: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]] = None
    prediction_intervals: Optional[Dict[str, List[float]]] = None
    timestamp: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    results: List[PredictionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    uptime_seconds: float
    mlflow_connected: bool
    models_available: int


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    model_name: str
    model_type: str
    version: str
    stage: str
    metrics: Dict[str, float]
    target_column: str
    repository: str


class RepositoryInfo(BaseModel):
    """Information about an available repository."""
    
    name: str
    data_weeks: int
    last_updated: Optional[str]
    available_metrics: List[str]


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests per API key
        self.minute_buckets: Dict[str, List[float]] = defaultdict(list)
        self.hour_buckets: Dict[str, List[float]] = defaultdict(list)
    
    def _clean_bucket(self, bucket: List[float], window_seconds: int) -> List[float]:
        """Remove expired entries from bucket."""
        now = time.time()
        return [t for t in bucket if now - t < window_seconds]
    
    def check_rate_limit(self, api_key: str) -> Tuple[bool, str]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, message).
        """
        now = time.time()
        
        # Clean and check minute bucket
        self.minute_buckets[api_key] = self._clean_bucket(
            self.minute_buckets[api_key], 60
        )
        
        if len(self.minute_buckets[api_key]) >= self.requests_per_minute:
            return False, "Rate limit exceeded: too many requests per minute"
        
        # Clean and check hour bucket
        self.hour_buckets[api_key] = self._clean_bucket(
            self.hour_buckets[api_key], 3600
        )
        
        if len(self.hour_buckets[api_key]) >= self.requests_per_hour:
            return False, "Rate limit exceeded: too many requests per hour"
        
        # Record this request
        self.minute_buckets[api_key].append(now)
        self.hour_buckets[api_key].append(now)
        
        return True, "OK"
    
    def get_remaining(self, api_key: str) -> Dict[str, int]:
        """Get remaining requests for an API key."""
        minute_remaining = self.requests_per_minute - len(self.minute_buckets.get(api_key, []))
        hour_remaining = self.requests_per_hour - len(self.hour_buckets.get(api_key, []))
        
        return {
            "minute_remaining": max(0, minute_remaining),
            "hour_remaining": max(0, hour_remaining)
        }


# =============================================================================
# API Key Management
# =============================================================================

class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self, keys_file: str = "api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys = self._load_keys()
    
    def _load_keys(self) -> Dict[str, Dict]:
        """Load API keys from file."""
        if self.keys_file.exists():
            import json
            with open(self.keys_file, "r") as f:
                return json.load(f)
        
        # Default keys for development
        return {
            "dev_key_12345": {
                "name": "Development",
                "tier": "free",
                "rate_limit_minute": 60,
                "rate_limit_hour": 1000
            }
        }
    
    def _save_keys(self):
        """Save keys to file."""
        import json
        with open(self.keys_file, "w") as f:
            json.dump(self.keys, f, indent=2)
    
    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return its info."""
        return self.keys.get(api_key)
    
    def generate_key(self, name: str, tier: str = "free") -> str:
        """Generate a new API key."""
        key = secrets.token_urlsafe(32)
        
        tier_limits = {
            "free": {"rate_limit_minute": 60, "rate_limit_hour": 1000},
            "pro": {"rate_limit_minute": 300, "rate_limit_hour": 10000},
            "enterprise": {"rate_limit_minute": 1000, "rate_limit_hour": 100000}
        }
        
        self.keys[key] = {
            "name": name,
            "tier": tier,
            **tier_limits.get(tier, tier_limits["free"]),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self._save_keys()
        return key


# =============================================================================
# Prediction Service
# =============================================================================

class PredictionService:
    """Service for handling predictions."""
    
    def __init__(self, data_dir: str = "repositories"):
        self.data_dir = Path(data_dir)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for a request."""
        key_str = f"{request.repository}_{request.target_column}_{request.horizon}_{request.model_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _check_cache(self, key: str) -> Optional[Dict]:
        """Check if result is cached."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                return entry["result"]
        return None
    
    def _cache_result(self, key: str, result: Dict):
        """Cache a result."""
        self._cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def get_repository_data(self, repository: str) -> pd.DataFrame:
        """Load repository data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.etl import load_repository_data
        
        repo_path = self.data_dir / repository
        
        if not repo_path.exists():
            raise ValueError(f"Repository not found: {repository}")
        
        return load_repository_data(str(repo_path))
    
    def predict(self, request: PredictionRequest) -> Dict:
        """Generate prediction for a request."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for {request.repository}")
            return cached
        
        # Load data
        df = self.get_repository_data(request.repository)
        
        if request.target_column not in df.columns:
            raise ValueError(f"Invalid target column: {request.target_column}")
        
        # Import prediction modules
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Use auto model selection if no model specified
        if request.model_type is None:
            from src.model_selection import auto_select_model
            
            result = auto_select_model(
                df=df,
                target_column=request.target_column,
                horizon=request.horizon,
                quick_mode=True
            )
            
            model_type = result.best_model_type
            predictions = [
                {"step": i+1, "prediction": p["y_pred"], "actual": p["y_actual"]}
                for i, p in enumerate(result.all_results[0].predictions) 
                if result.all_results[0].model_type == model_type
            ]
            
            # Get predictions from best result
            for r in result.all_results:
                if r.model_type == model_type:
                    predictions = [
                        {"step": p["step"], "prediction": p["y_pred"]}
                        for p in r.predictions
                    ]
                    break
            
            intervals = result.prediction_intervals
            
        else:
            # Use specific model
            from src.model_engine import train_predict_recursive
            
            result = train_predict_recursive(
                df=df,
                target_column=request.target_column,
                model_type=request.model_type,
                horizon=request.horizon,
                repo_name=request.repository
            )
            
            model_type = request.model_type
            predictions = [
                {
                    "step": int(row["step"]),
                    "date": row["ds"].isoformat() if hasattr(row["ds"], 'isoformat') else str(row["ds"]),
                    "prediction": float(row["y_pred"]),
                    "actual": float(row["y_actual"])
                }
                for _, row in result["predictions"].iterrows()
            ]
            
            intervals = None  # TODO: Add intervals for single model
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "repository": request.repository,
            "target_column": request.target_column,
            "model_type": model_type,
            "predictions": predictions,
            "prediction_intervals": intervals if request.include_intervals else None,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        }
        
        # Cache result
        self._cache_result(cache_key, response)
        
        return response
    
    def list_repositories(self) -> List[Dict]:
        """List available repositories."""
        repos = []
        
        for repo_dir in self.data_dir.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith("."):
                try:
                    df = self.get_repository_data(repo_dir.name)
                    repos.append({
                        "name": repo_dir.name,
                        "data_weeks": len(df),
                        "available_metrics": ["commits", "new_stars", "issues_opened", "prs_opened"]
                    })
                except Exception:
                    pass
        
        return repos


# =============================================================================
# FastAPI Application
# =============================================================================

# Initialize services
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
prediction_service = PredictionService()
start_time = time.time()

# Create FastAPI app
app = FastAPI(
    title="GitHub Activity Predictor API",
    description="REST API for predicting GitHub repository activity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> Dict:
    """Verify API key and check rate limits."""
    if not api_key:
        # Allow unauthenticated requests with strict limits
        api_key = "anonymous"
    
    # Check rate limit
    allowed, message = rate_limiter.check_rate_limit(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=message)
    
    # Validate key (skip for anonymous)
    if api_key != "anonymous":
        key_info = api_key_manager.validate_key(api_key)
        if not key_info:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return key_info
    
    return {"name": "anonymous", "tier": "free"}


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.search_experiments()
        mlflow_connected = True
    except Exception:
        mlflow_connected = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
        mlflow_connected=mlflow_connected,
        models_available=len(prediction_service.list_repositories())
    )


@app.get("/repositories", response_model=List[RepositoryInfo])
async def list_repositories(
    limit: int = Query(default=100, le=1000),
    api_key_info: Dict = Depends(verify_api_key)
):
    """List available repositories."""
    repos = prediction_service.list_repositories()
    
    return [
        RepositoryInfo(
            name=r["name"],
            data_weeks=r["data_weeks"],
            last_updated=None,
            available_metrics=r["available_metrics"]
        )
        for r in repos[:limit]
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key_info: Dict = Depends(verify_api_key)
):
    """Generate prediction for a repository."""
    try:
        result = prediction_service.predict(request)
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    api_key_info: Dict = Depends(verify_api_key)
):
    """Generate predictions for multiple repositories."""
    start = time.time()
    results = []
    
    for pred_request in request.requests:
        try:
            result = prediction_service.predict(pred_request)
            results.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Batch prediction error for {pred_request.repository}: {e}")
    
    return BatchPredictionResponse(
        results=results,
        total_processing_time_ms=(time.time() - start) * 1000
    )


@app.get("/rate-limit")
async def get_rate_limit_status(
    api_key: str = Depends(api_key_header)
):
    """Get rate limit status for current API key."""
    if not api_key:
        api_key = "anonymous"
    
    remaining = rate_limiter.get_remaining(api_key)
    
    return {
        "api_key": api_key[:8] + "..." if len(api_key) > 8 else api_key,
        **remaining
    }


@app.post("/api-key/generate")
async def generate_api_key(
    name: str = Query(..., description="Name for the API key"),
    tier: str = Query(default="free", description="Tier (free, pro, enterprise)"),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    """Generate a new API key (admin only)."""
    # Simple admin key check
    expected_admin_key = os.environ.get("ADMIN_API_KEY", "admin_secret_key")
    
    if admin_key != expected_admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    new_key = api_key_manager.generate_key(name, tier)
    
    return {
        "api_key": new_key,
        "name": name,
        "tier": tier,
        "message": "Store this key securely. It will not be shown again."
    }


# =============================================================================
# Main
# =============================================================================

def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Start the API server."""
    uvicorn.run(
        "src.api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Activity Predictor API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    start_api_server(args.host, args.port, args.reload)
