"""
ETL Module - GitHub Activity Predictor
======================================

This module handles the extraction, transformation, and loading of GitHub
repository data from CSV files into clean weekly time series DataFrames.

Features:
- Robust handling of missing files
- UTC datetime conversion
- Weekly resampling (W frequency)
- Sparsity handling (NaN -> 0)
- Merged DataFrame with columns: ds, commits, new_stars, issues_opened, prs_opened
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ETLProcessor:
    """
    ETL Processor for GitHub repository data.
    
    Transforms raw CSV files (commits.csv, issues.csv, pull_requests.csv, stargazers.csv)
    into a unified weekly time series DataFrame.
    """
    
    # Expected CSV files and their date columns
    FILE_CONFIG = {
        "commits.csv": {
            "date_column": "author_date",
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
    
    def __init__(self, repo_path: str):
        """
        Initialize the ETL processor.
        
        Args:
            repo_path: Path to the repository folder containing CSV files.
        """
        self.repo_path = Path(repo_path)
        self.repo_name = self.repo_path.name
        
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        
        logger.info(f"Initialized ETL processor for repository: {self.repo_name}")
    
    def _read_csv_safe(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Safely read a CSV file, returning None if it doesn't exist.
        
        Args:
            filename: Name of the CSV file to read.
            
        Returns:
            DataFrame or None if file doesn't exist.
        """
        file_path = self.repo_path / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {filename}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return None
    
    def _convert_to_datetime_utc(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Convert a date column to UTC datetime.
        
        Args:
            df: Input DataFrame.
            date_column: Name of the column to convert.
            
        Returns:
            DataFrame with converted datetime column.
        """
        if df is None or date_column not in df.columns:
            return df
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], utc=True, errors="coerce")
        
        # Drop rows with invalid dates
        invalid_count = df[date_column].isna().sum()
        if invalid_count > 0:
            logger.warning(f"Dropped {invalid_count} rows with invalid dates in {date_column}")
            df = df.dropna(subset=[date_column])
        
        return df
    
    def _resample_weekly(
        self, 
        df: pd.DataFrame, 
        date_column: str, 
        metric_name: str
    ) -> pd.DataFrame:
        """
        Resample data to weekly frequency by counting events.
        
        Args:
            df: Input DataFrame with datetime column.
            date_column: Name of the datetime column.
            metric_name: Name for the output metric column.
            
        Returns:
            DataFrame with columns: ds (week end date), metric_name (count).
        """
        if df is None or len(df) == 0:
            logger.warning(f"Empty DataFrame for metric {metric_name}")
            return pd.DataFrame(columns=["ds", metric_name])
        
        df = df.copy()
        df = df.set_index(date_column)
        
        # Resample to weekly frequency (Sunday end)
        weekly_counts = df.resample("W").size()
        
        result = pd.DataFrame({
            "ds": weekly_counts.index,
            metric_name: weekly_counts.values
        })
        
        # Remove timezone info from ds for compatibility
        result["ds"] = result["ds"].dt.tz_localize(None)
        
        logger.info(f"Resampled {metric_name}: {len(result)} weeks")
        return result
    
    def _process_single_file(self, filename: str, config: Dict) -> pd.DataFrame:
        """
        Process a single CSV file through the ETL pipeline.
        
        Args:
            filename: Name of the CSV file.
            config: Configuration dict with date_column and metric_name.
            
        Returns:
            Resampled weekly DataFrame.
        """
        date_column = config["date_column"]
        metric_name = config["metric_name"]
        
        # Read CSV
        df = self._read_csv_safe(filename)
        
        if df is None:
            return pd.DataFrame(columns=["ds", metric_name])
        
        # Convert datetime
        df = self._convert_to_datetime_utc(df, date_column)
        
        # Resample weekly
        weekly_df = self._resample_weekly(df, date_column, metric_name)
        
        return weekly_df
    
    def extract_and_transform(self) -> pd.DataFrame:
        """
        Main ETL pipeline: Extract data from CSV files and transform to weekly time series.
        
        Returns:
            Clean DataFrame with columns: ds, commits, new_stars, issues_opened, prs_opened.
            All NaN values are replaced with 0.
        """
        logger.info(f"Starting ETL for repository: {self.repo_name}")
        
        # Process each file
        weekly_dataframes = {}
        for filename, config in self.FILE_CONFIG.items():
            metric_name = config["metric_name"]
            weekly_df = self._process_single_file(filename, config)
            weekly_dataframes[metric_name] = weekly_df
        
        # Merge all weekly DataFrames
        merged_df = self._merge_weekly_data(weekly_dataframes)
        
        # Fill NaN with 0 (sparsity handling)
        merged_df = merged_df.fillna(0)
        
        # Ensure integer types for count columns
        count_columns = ["commits", "new_stars", "issues_opened", "prs_opened"]
        for col in count_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].astype(int)
        
        # Sort by date
        merged_df = merged_df.sort_values("ds").reset_index(drop=True)
        
        logger.info(
            f"ETL complete for {self.repo_name}: "
            f"{len(merged_df)} weeks, date range: "
            f"{merged_df['ds'].min()} to {merged_df['ds'].max()}"
        )
        
        return merged_df
    
    def _merge_weekly_data(self, weekly_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple weekly DataFrames on the ds (date) column.
        
        Args:
            weekly_dataframes: Dict mapping metric names to their DataFrames.
            
        Returns:
            Merged DataFrame with all metrics.
        """
        # Get all unique dates across all DataFrames
        all_dates = set()
        for metric_name, df in weekly_dataframes.items():
            if len(df) > 0:
                all_dates.update(df["ds"].tolist())
        
        if not all_dates:
            logger.warning("No data found in any file")
            return pd.DataFrame(columns=["ds", "commits", "new_stars", "issues_opened", "prs_opened"])
        
        # Create base DataFrame with all dates
        merged_df = pd.DataFrame({"ds": sorted(all_dates)})
        
        # Merge each metric
        for metric_name, df in weekly_dataframes.items():
            if len(df) > 0:
                merged_df = merged_df.merge(
                    df[["ds", metric_name]], 
                    on="ds", 
                    how="left"
                )
            else:
                merged_df[metric_name] = 0
        
        return merged_df


def load_repository_data(repo_path: str) -> pd.DataFrame:
    """
    Convenience function to load and transform repository data.
    
    Args:
        repo_path: Path to the repository folder.
        
    Returns:
        Clean weekly time series DataFrame.
    """
    processor = ETLProcessor(repo_path)
    return processor.extract_and_transform()


def list_available_repositories(data_dir: str) -> List[str]:
    """
    List all available repositories in the data directory.
    
    Args:
        data_dir: Path to the data/repositories folder.
        
    Returns:
        List of repository folder names.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []
    
    repos = [
        d.name for d in data_path.iterdir() 
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    return sorted(repos)


def get_repository_summary(repo_path: str) -> Dict:
    """
    Get a summary of repository data.
    
    Args:
        repo_path: Path to the repository folder.
        
    Returns:
        Dictionary with summary statistics.
    """
    try:
        df = load_repository_data(repo_path)
        
        summary = {
            "repo_name": Path(repo_path).name,
            "total_weeks": len(df),
            "date_range": {
                "start": df["ds"].min().isoformat() if len(df) > 0 else None,
                "end": df["ds"].max().isoformat() if len(df) > 0 else None
            },
            "totals": {
                "commits": int(df["commits"].sum()),
                "new_stars": int(df["new_stars"].sum()),
                "issues_opened": int(df["issues_opened"].sum()),
                "prs_opened": int(df["prs_opened"].sum())
            },
            "averages": {
                "commits_per_week": float(df["commits"].mean()),
                "new_stars_per_week": float(df["new_stars"].mean()),
                "issues_per_week": float(df["issues_opened"].mean()),
                "prs_per_week": float(df["prs_opened"].mean())
            }
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error getting repository summary: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        # Default test path
        repo_path = "repositories/3b1b__manim"
    
    print(f"\n{'='*60}")
    print(f"ETL Pipeline Test - Repository: {repo_path}")
    print(f"{'='*60}\n")
    
    try:
        df = load_repository_data(repo_path)
        print("Resulting DataFrame:")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst 10 rows:\n{df.head(10)}")
        print(f"\nLast 10 rows:\n{df.tail(10)}")
        print(f"\nStatistics:\n{df.describe()}")
        
        print("\n\nRepository Summary:")
        summary = get_repository_summary(repo_path)
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
