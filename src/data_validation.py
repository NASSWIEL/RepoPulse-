"""
Data Validation Module - GitHub Activity Predictor
===================================================

This module implements data quality checks for the ML pipeline,
including schema validation, anomaly detection, and data drift monitoring.

Features:
- Schema validation against expected data structure
- Statistical anomaly detection (IQR, Z-score methods)
- Data drift monitoring using statistical tests
- Missing value detection and reporting
- Data quality scoring
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    check_name: str
    passed: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    
    dataset_name: str
    validation_results: List[ValidationResult]
    overall_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def passed(self) -> bool:
        """Check if all critical validations passed."""
        return not any(
            r.severity == "error" and not r.passed 
            for r in self.validation_results
        )
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level failures."""
        return [r for r in self.validation_results if not r.passed and r.severity == "error"]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warning-level issues."""
        return [r for r in self.validation_results if not r.passed and r.severity == "warning"]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "validation_results": [r.to_dict() for r in self.validation_results],
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SchemaValidator:
    """Validates data schema against expected structure."""
    
    # Expected schema for the weekly time series data
    EXPECTED_SCHEMA = {
        "ds": {"dtype": "datetime64", "nullable": False},
        "commits": {"dtype": "int64", "nullable": False, "min": 0},
        "new_stars": {"dtype": "int64", "nullable": False, "min": 0},
        "issues_opened": {"dtype": "int64", "nullable": False, "min": 0},
        "prs_opened": {"dtype": "int64", "nullable": False, "min": 0}
    }
    
    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize schema validator.
        
        Args:
            schema: Custom schema. If None, uses default.
        """
        self.schema = schema or self.EXPECTED_SCHEMA
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            List of validation results.
        """
        results = []
        
        # Check required columns
        missing_cols = set(self.schema.keys()) - set(df.columns)
        if missing_cols:
            results.append(ValidationResult(
                check_name="required_columns",
                passed=False,
                severity="error",
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": list(missing_cols)}
            ))
        else:
            results.append(ValidationResult(
                check_name="required_columns",
                passed=True,
                severity="info",
                message="All required columns present"
            ))
        
        # Check each column
        for col_name, col_spec in self.schema.items():
            if col_name not in df.columns:
                continue
            
            # Check dtype
            if "datetime" in col_spec.get("dtype", ""):
                is_datetime = pd.api.types.is_datetime64_any_dtype(df[col_name])
                if not is_datetime:
                    results.append(ValidationResult(
                        check_name=f"dtype_{col_name}",
                        passed=False,
                        severity="error",
                        message=f"Column {col_name} should be datetime",
                        details={"actual_dtype": str(df[col_name].dtype)}
                    ))
            elif "int" in col_spec.get("dtype", ""):
                is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
                if not is_numeric:
                    results.append(ValidationResult(
                        check_name=f"dtype_{col_name}",
                        passed=False,
                        severity="error",
                        message=f"Column {col_name} should be numeric",
                        details={"actual_dtype": str(df[col_name].dtype)}
                    ))
            
            # Check nullability
            if not col_spec.get("nullable", True):
                null_count = df[col_name].isna().sum()
                if null_count > 0:
                    results.append(ValidationResult(
                        check_name=f"nullable_{col_name}",
                        passed=False,
                        severity="error",
                        message=f"Column {col_name} has {null_count} null values",
                        details={"null_count": int(null_count)}
                    ))
            
            # Check min/max constraints
            if "min" in col_spec and pd.api.types.is_numeric_dtype(df[col_name]):
                min_val = df[col_name].min()
                if min_val < col_spec["min"]:
                    results.append(ValidationResult(
                        check_name=f"min_value_{col_name}",
                        passed=False,
                        severity="warning",
                        message=f"Column {col_name} has values below minimum {col_spec['min']}",
                        details={"actual_min": float(min_val)}
                    ))
        
        return results


class AnomalyDetector:
    """Detects anomalies in time series data."""
    
    def __init__(
        self, 
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0
    ):
        """
        Initialize anomaly detector.
        
        Args:
            iqr_multiplier: Multiplier for IQR-based detection.
            z_score_threshold: Threshold for Z-score based detection.
        """
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
    
    def detect_iqr_anomalies(
        self, 
        series: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect anomalies using IQR method.
        
        Args:
            series: Numeric series to check.
            
        Returns:
            Tuple of (boolean mask of anomalies, statistics dict).
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        
        anomalies = (series < lower_bound) | (series > upper_bound)
        
        stats = {
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "n_anomalies": int(anomalies.sum()),
            "anomaly_rate": float(anomalies.mean())
        }
        
        return anomalies.values, stats
    
    def detect_zscore_anomalies(
        self, 
        series: pd.Series
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            series: Numeric series to check.
            
        Returns:
            Tuple of (boolean mask of anomalies, statistics dict).
        """
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return np.zeros(len(series), dtype=bool), {
                "mean": float(mean),
                "std": 0.0,
                "n_anomalies": 0,
                "anomaly_rate": 0.0
            }
        
        z_scores = np.abs((series - mean) / std)
        anomalies = z_scores > self.z_score_threshold
        
        stats = {
            "mean": float(mean),
            "std": float(std),
            "threshold": self.z_score_threshold,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_rate": float(anomalies.mean())
        }
        
        return anomalies.values, stats
    
    def validate_column(
        self, 
        series: pd.Series, 
        column_name: str
    ) -> List[ValidationResult]:
        """
        Run anomaly detection on a column.
        
        Args:
            series: Column data.
            column_name: Name of the column.
            
        Returns:
            List of validation results.
        """
        results = []
        
        if not pd.api.types.is_numeric_dtype(series):
            return results
        
        # IQR method
        iqr_anomalies, iqr_stats = self.detect_iqr_anomalies(series)
        
        anomaly_rate = iqr_stats["anomaly_rate"]
        if anomaly_rate > 0.1:  # More than 10% anomalies
            severity = "warning" if anomaly_rate < 0.2 else "error"
            results.append(ValidationResult(
                check_name=f"iqr_anomalies_{column_name}",
                passed=False,
                severity=severity,
                message=f"Column {column_name} has {iqr_stats['n_anomalies']} IQR anomalies ({anomaly_rate:.1%})",
                details=iqr_stats
            ))
        else:
            results.append(ValidationResult(
                check_name=f"iqr_anomalies_{column_name}",
                passed=True,
                severity="info",
                message=f"Column {column_name} passed IQR anomaly check",
                details=iqr_stats
            ))
        
        # Z-score method
        z_anomalies, z_stats = self.detect_zscore_anomalies(series)
        
        if z_stats["n_anomalies"] > 5:
            results.append(ValidationResult(
                check_name=f"zscore_anomalies_{column_name}",
                passed=False,
                severity="warning",
                message=f"Column {column_name} has {z_stats['n_anomalies']} Z-score anomalies",
                details=z_stats
            ))
        
        return results


class DataDriftDetector:
    """Detects data drift between reference and current data."""
    
    def __init__(
        self, 
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.2
    ):
        """
        Initialize drift detector.
        
        Args:
            p_value_threshold: P-value threshold for statistical tests.
            psi_threshold: Population Stability Index threshold.
        """
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
    
    def ks_test(
        self, 
        reference: pd.Series, 
        current: pd.Series
    ) -> Tuple[float, float, bool]:
        """
        Perform Kolmogorov-Smirnov test for drift.
        
        Args:
            reference: Reference (baseline) data.
            current: Current data to test.
            
        Returns:
            Tuple of (KS statistic, p-value, is_drifted).
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        is_drifted = p_value < self.p_value_threshold
        return float(statistic), float(p_value), is_drifted
    
    def calculate_psi(
        self, 
        reference: pd.Series, 
        current: pd.Series,
        n_bins: int = 10
    ) -> Tuple[float, bool]:
        """
        Calculate Population Stability Index.
        
        Args:
            reference: Reference data.
            current: Current data.
            n_bins: Number of bins for discretization.
            
        Returns:
            Tuple of (PSI value, is_drifted).
        """
        # Create bins from reference data
        _, bins = pd.cut(reference, bins=n_bins, retbins=True)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Calculate distributions
        ref_counts = pd.cut(reference, bins=bins).value_counts(normalize=True).sort_index()
        cur_counts = pd.cut(current, bins=bins).value_counts(normalize=True).sort_index()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = ref_counts.values + eps
        cur_pct = cur_counts.values + eps
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        is_drifted = psi > self.psi_threshold
        return float(psi), is_drifted
    
    def detect_drift(
        self, 
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            reference_df: Reference (baseline) DataFrame.
            current_df: Current DataFrame to check.
            columns: Columns to check. If None, checks all numeric columns.
            
        Returns:
            List of validation results.
        """
        results = []
        
        if columns is None:
            columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            
            ref_data = reference_df[col].dropna()
            cur_data = current_df[col].dropna()
            
            if len(ref_data) < 10 or len(cur_data) < 10:
                continue
            
            # KS Test
            ks_stat, p_value, ks_drifted = self.ks_test(ref_data, cur_data)
            
            if ks_drifted:
                results.append(ValidationResult(
                    check_name=f"ks_drift_{col}",
                    passed=False,
                    severity="warning",
                    message=f"Column {col} shows statistical drift (KS test p={p_value:.4f})",
                    details={
                        "ks_statistic": ks_stat,
                        "p_value": p_value,
                        "threshold": self.p_value_threshold
                    }
                ))
            
            # PSI
            try:
                psi_value, psi_drifted = self.calculate_psi(ref_data, cur_data)
                
                if psi_drifted:
                    results.append(ValidationResult(
                        check_name=f"psi_drift_{col}",
                        passed=False,
                        severity="warning",
                        message=f"Column {col} shows distribution drift (PSI={psi_value:.4f})",
                        details={
                            "psi": psi_value,
                            "threshold": self.psi_threshold
                        }
                    ))
            except Exception:
                pass
        
        if not results:
            results.append(ValidationResult(
                check_name="drift_check",
                passed=True,
                severity="info",
                message="No significant data drift detected"
            ))
        
        return results


class DataQualityValidator:
    """Main validator that combines all validation checks."""
    
    def __init__(
        self,
        schema: Optional[Dict] = None,
        reference_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize data quality validator.
        
        Args:
            schema: Custom schema for validation.
            reference_data: Reference data for drift detection.
        """
        self.schema_validator = SchemaValidator(schema)
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DataDriftDetector()
        self.reference_data = reference_data
    
    def validate(
        self, 
        df: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> DataQualityReport:
        """
        Run all validation checks on a DataFrame.
        
        Args:
            df: DataFrame to validate.
            dataset_name: Name of the dataset for reporting.
            
        Returns:
            DataQualityReport with all results.
        """
        results = []
        
        # Basic checks
        results.append(ValidationResult(
            check_name="row_count",
            passed=len(df) > 0,
            severity="error" if len(df) == 0 else "info",
            message=f"Dataset has {len(df)} rows",
            details={"row_count": len(df)}
        ))
        
        # Schema validation
        schema_results = self.schema_validator.validate(df)
        results.extend(schema_results)
        
        # Anomaly detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            anomaly_results = self.anomaly_detector.validate_column(df[col], col)
            results.extend(anomaly_results)
        
        # Drift detection if reference data is available
        if self.reference_data is not None:
            drift_results = self.drift_detector.detect_drift(
                self.reference_data, df, list(numeric_cols)
            )
            results.extend(drift_results)
        
        # Calculate overall score
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        
        # Weight by severity
        error_failures = sum(1 for r in results if not r.passed and r.severity == "error")
        warning_failures = sum(1 for r in results if not r.passed and r.severity == "warning")
        
        # Score: 100 - (error_failures * 20) - (warning_failures * 5)
        score = max(0, min(100, 100 - error_failures * 20 - warning_failures * 5))
        
        return DataQualityReport(
            dataset_name=dataset_name,
            validation_results=results,
            overall_score=score
        )
    
    def set_reference_data(self, df: pd.DataFrame):
        """Set reference data for drift detection."""
        self.reference_data = df.copy()
    
    def validate_for_training(
        self, 
        df: pd.DataFrame,
        min_rows: int = 52,
        target_column: str = "commits"
    ) -> Tuple[bool, DataQualityReport]:
        """
        Validate data is suitable for model training.
        
        Args:
            df: DataFrame to validate.
            min_rows: Minimum required rows.
            target_column: Target column for prediction.
            
        Returns:
            Tuple of (is_valid, report).
        """
        results = []
        
        # Check minimum data
        if len(df) < min_rows:
            results.append(ValidationResult(
                check_name="min_training_data",
                passed=False,
                severity="error",
                message=f"Insufficient data: {len(df)} rows (need {min_rows})",
                details={"row_count": len(df), "min_required": min_rows}
            ))
        else:
            results.append(ValidationResult(
                check_name="min_training_data",
                passed=True,
                severity="info",
                message=f"Sufficient training data: {len(df)} rows"
            ))
        
        # Check target column
        if target_column not in df.columns:
            results.append(ValidationResult(
                check_name="target_column",
                passed=False,
                severity="error",
                message=f"Target column '{target_column}' not found"
            ))
        else:
            # Check target variance
            target_variance = df[target_column].var()
            if target_variance == 0:
                results.append(ValidationResult(
                    check_name="target_variance",
                    passed=False,
                    severity="error",
                    message=f"Target column '{target_column}' has zero variance"
                ))
            else:
                results.append(ValidationResult(
                    check_name="target_variance",
                    passed=True,
                    severity="info",
                    message=f"Target column has sufficient variance: {target_variance:.2f}"
                ))
        
        # Check for date gaps
        if "ds" in df.columns:
            df_sorted = df.sort_values("ds")
            date_diffs = df_sorted["ds"].diff().dropna()
            
            if pd.api.types.is_datetime64_any_dtype(df["ds"]):
                # Check for gaps > 2 weeks
                gaps = date_diffs[date_diffs > pd.Timedelta(days=14)]
                if len(gaps) > 0:
                    results.append(ValidationResult(
                        check_name="date_gaps",
                        passed=False,
                        severity="warning",
                        message=f"Found {len(gaps)} time gaps > 2 weeks",
                        details={"n_gaps": len(gaps)}
                    ))
        
        # Standard validation
        standard_report = self.validate(df, "training_data")
        results.extend(standard_report.validation_results)
        
        # Calculate score
        error_count = sum(1 for r in results if not r.passed and r.severity == "error")
        score = max(0, 100 - error_count * 20)
        
        report = DataQualityReport(
            dataset_name="training_data",
            validation_results=results,
            overall_score=score
        )
        
        return report.passed, report


def create_validator(
    reference_data: Optional[pd.DataFrame] = None
) -> DataQualityValidator:
    """
    Factory function to create a data quality validator.
    
    Args:
        reference_data: Optional reference data for drift detection.
        
    Returns:
        Configured DataQualityValidator instance.
    """
    return DataQualityValidator(reference_data=reference_data)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.etl import load_repository_data
    
    print("Testing Data Validation Module")
    print("=" * 60)
    
    repo_path = "repositories/3b1b__manim"
    
    try:
        df = load_repository_data(repo_path)
        print(f"Loaded data: {len(df)} rows")
        
        # Create validator
        validator = create_validator()
        
        # Run validation
        report = validator.validate(df, "3b1b__manim")
        
        print(f"\n{'='*60}")
        print(f"Data Quality Report: {report.dataset_name}")
        print(f"{'='*60}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Passed: {report.passed}")
        print(f"Errors: {len(report.errors)}")
        print(f"Warnings: {len(report.warnings)}")
        
        print("\nValidation Results:")
        for result in report.validation_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] [{result.severity.upper()}] {result.check_name}: {result.message}")
        
        # Training validation
        print("\n" + "=" * 60)
        print("Training Validation")
        print("=" * 60)
        
        is_valid, train_report = validator.validate_for_training(df)
        print(f"Valid for training: {is_valid}")
        print(f"Score: {train_report.overall_score:.1f}/100")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
