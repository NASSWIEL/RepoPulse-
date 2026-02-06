"""
Data Ingestion Module - GitHub Activity Predictor
==================================================

This module handles automated data fetching from the GitHub API.
It fetches commits, issues, pull requests, and stargazers for repositories.

Features:
- Automatic data fetching using GitHub API
- Incremental updates (only fetch new data)
- Rate limiting handling
- Repository tracking to avoid duplicates
- Concurrent fetching for efficiency
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GitHubAPIClient:
    """
    Client for interacting with the GitHub API.
    Handles authentication, rate limiting, and pagination.
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: str, max_retries: int = 3, timeout: int = 30):
        """
        Initialize the GitHub API client.
        
        Args:
            token: GitHub personal access token.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        self.token = token
        self.timeout = timeout
        self.session = self._create_session(max_retries)
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        
        logger.info("GitHub API client initialized")
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Activity-Predictor"
        })
        
        return session
    
    def _handle_rate_limit(self, response: requests.Response):
        """Handle rate limiting from GitHub API."""
        self._rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        self._rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
        
        if self._rate_limit_remaining < 10:
            wait_time = max(0, self._rate_limit_reset - time.time()) + 1
            logger.warning(f"Rate limit low ({self._rate_limit_remaining}). Waiting {wait_time:.0f}s")
            time.sleep(wait_time)
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None
    ) -> Tuple[Optional[Any], bool]:
        """
        Make a request to the GitHub API.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            
        Returns:
            Tuple of (response data, success flag).
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            self._handle_rate_limit(response)
            
            if response.status_code == 200:
                return response.json(), True
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {endpoint}")
                return None, False
            elif response.status_code == 403:
                logger.error(f"Access forbidden: {endpoint}")
                return None, False
            else:
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return None, False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None, False
    
    def get_paginated_data(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        max_pages: int = 100,
        per_page: int = 100
    ) -> List[Dict]:
        """
        Fetch all pages of data from a paginated endpoint.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            max_pages: Maximum number of pages to fetch.
            per_page: Items per page.
            
        Returns:
            List of all items from all pages.
        """
        params = params or {}
        params["per_page"] = per_page
        
        all_data = []
        page = 1
        
        while page <= max_pages:
            params["page"] = page
            data, success = self._make_request(endpoint, params)
            
            if not success or not data:
                break
            
            all_data.extend(data)
            
            if len(data) < per_page:
                break
            
            page += 1
            logger.debug(f"Fetched page {page - 1}, total items: {len(all_data)}")
        
        return all_data
    
    def get_repository_info(self, owner: str, repo: str) -> Optional[Dict]:
        """Get basic repository information."""
        endpoint = f"/repos/{owner}/{repo}"
        data, success = self._make_request(endpoint)
        return data if success else None
    
    def get_commits(
        self, 
        owner: str, 
        repo: str, 
        since: Optional[str] = None,
        until: Optional[str] = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """
        Fetch commits for a repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            since: ISO 8601 date to fetch commits after.
            until: ISO 8601 date to fetch commits before.
            max_pages: Maximum pages to fetch.
            
        Returns:
            List of commit objects.
        """
        endpoint = f"/repos/{owner}/{repo}/commits"
        params = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        
        return self.get_paginated_data(endpoint, params, max_pages)
    
    def get_issues(
        self, 
        owner: str, 
        repo: str,
        state: str = "all",
        since: Optional[str] = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """
        Fetch issues for a repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            state: Issue state filter (all, open, closed).
            since: ISO 8601 date to fetch issues updated after.
            max_pages: Maximum pages to fetch.
            
        Returns:
            List of issue objects (excluding pull requests).
        """
        endpoint = f"/repos/{owner}/{repo}/issues"
        params = {"state": state, "sort": "created", "direction": "asc"}
        if since:
            params["since"] = since
        
        all_issues = self.get_paginated_data(endpoint, params, max_pages)
        
        # Filter out pull requests (they come through the issues endpoint too)
        return [issue for issue in all_issues if "pull_request" not in issue]
    
    def get_pull_requests(
        self, 
        owner: str, 
        repo: str,
        state: str = "all",
        max_pages: int = 100
    ) -> List[Dict]:
        """
        Fetch pull requests for a repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            state: PR state filter (all, open, closed).
            max_pages: Maximum pages to fetch.
            
        Returns:
            List of pull request objects.
        """
        endpoint = f"/repos/{owner}/{repo}/pulls"
        params = {"state": state, "sort": "created", "direction": "asc"}
        
        return self.get_paginated_data(endpoint, params, max_pages)
    
    def get_stargazers(
        self, 
        owner: str, 
        repo: str,
        max_pages: int = 100
    ) -> List[Dict]:
        """
        Fetch stargazers with timestamp for a repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            max_pages: Maximum pages to fetch.
            
        Returns:
            List of stargazer objects with starred_at timestamp.
        """
        endpoint = f"/repos/{owner}/{repo}/stargazers"
        
        # Need special header for starred_at timestamp
        old_accept = self.session.headers.get("Accept")
        self.session.headers["Accept"] = "application/vnd.github.v3.star+json"
        
        stargazers = self.get_paginated_data(endpoint, max_pages=max_pages)
        
        self.session.headers["Accept"] = old_accept
        
        return stargazers
    
    def check_token_validity(self) -> bool:
        """Check if the token is valid."""
        data, success = self._make_request("/user")
        if success:
            logger.info(f"Authenticated as: {data.get('login', 'unknown')}")
            return True
        return False


class RepositoryTracker:
    """
    Tracks fetched repositories to avoid duplicate downloads.
    Stores metadata about last fetch times and data hashes.
    """
    
    def __init__(self, tracking_file: str = "fetch_tracking.json"):
        """
        Initialize the repository tracker.
        
        Args:
            tracking_file: Path to the JSON file storing tracking data.
        """
        self.tracking_file = Path(tracking_file)
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading tracking file: {e}")
        return {"repositories": {}, "last_updated": None}
    
    def _save_tracking_data(self):
        """Save tracking data to file."""
        self.tracking_data["last_updated"] = datetime.utcnow().isoformat()
        try:
            with open(self.tracking_file, "w") as f:
                json.dump(self.tracking_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tracking file: {e}")
    
    def is_repository_tracked(self, full_name: str) -> bool:
        """Check if a repository has been fetched before."""
        return full_name in self.tracking_data["repositories"]
    
    def get_last_fetch_time(self, full_name: str) -> Optional[str]:
        """Get the last fetch timestamp for a repository."""
        repo_data = self.tracking_data["repositories"].get(full_name, {})
        return repo_data.get("last_fetch")
    
    def update_repository_tracking(
        self, 
        full_name: str, 
        data_hashes: Dict[str, str],
        record_counts: Dict[str, int]
    ):
        """
        Update tracking data for a repository.
        
        Args:
            full_name: Full repository name (owner/repo).
            data_hashes: Dict mapping data type to content hash.
            record_counts: Dict mapping data type to record count.
        """
        self.tracking_data["repositories"][full_name] = {
            "last_fetch": datetime.utcnow().isoformat(),
            "data_hashes": data_hashes,
            "record_counts": record_counts
        }
        self._save_tracking_data()
    
    def should_update(
        self, 
        full_name: str, 
        update_interval_hours: int = 24
    ) -> bool:
        """
        Check if a repository should be updated based on time interval.
        
        Args:
            full_name: Full repository name.
            update_interval_hours: Minimum hours between updates.
            
        Returns:
            True if repository should be updated.
        """
        last_fetch = self.get_last_fetch_time(full_name)
        if not last_fetch:
            return True
        
        try:
            last_fetch_dt = datetime.fromisoformat(last_fetch.replace("Z", "+00:00"))
            threshold = datetime.utcnow() - timedelta(hours=update_interval_hours)
            return last_fetch_dt.replace(tzinfo=None) < threshold
        except Exception:
            return True


class DataIngestionPipeline:
    """
    Main pipeline for ingesting GitHub repository data.
    Handles fetching, transformation, and storage of data.
    """
    
    def __init__(
        self, 
        github_token: str,
        data_dir: str = "repositories",
        tracking_file: str = "fetch_tracking.json"
    ):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            github_token: GitHub personal access token.
            data_dir: Directory to store repository data.
            tracking_file: Path to the tracking file.
        """
        self.client = GitHubAPIClient(github_token)
        self.tracker = RepositoryTracker(tracking_file)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data ingestion pipeline initialized. Data dir: {self.data_dir}")
    
    def _compute_data_hash(self, data: List[Dict]) -> str:
        """Compute a hash of the data for change detection."""
        if not data:
            return "empty"
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _transform_commits(self, commits: List[Dict]) -> pd.DataFrame:
        """Transform raw commit data to DataFrame."""
        if not commits:
            return pd.DataFrame(columns=[
                "sha", "author_login", "author_name", "author_email", "author_date",
                "committer_login", "committer_name", "committer_email", "committer_date",
                "message", "comment_count", "url", "fetched_at"
            ])
        
        records = []
        for commit in commits:
            commit_data = commit.get("commit", {})
            author = commit_data.get("author", {}) or {}
            committer = commit_data.get("committer", {}) or {}
            
            records.append({
                "sha": commit.get("sha", ""),
                "author_login": (commit.get("author") or {}).get("login", ""),
                "author_name": author.get("name", ""),
                "author_email": author.get("email", ""),
                "author_date": author.get("date", ""),
                "committer_login": (commit.get("committer") or {}).get("login", ""),
                "committer_name": committer.get("name", ""),
                "committer_email": committer.get("email", ""),
                "committer_date": committer.get("date", ""),
                "message": commit_data.get("message", "")[:500],  # Truncate long messages
                "comment_count": commit_data.get("comment_count", 0),
                "url": commit.get("html_url", ""),
                "fetched_at": datetime.utcnow().isoformat()
            })
        
        return pd.DataFrame(records)
    
    def _transform_issues(self, issues: List[Dict]) -> pd.DataFrame:
        """Transform raw issue data to DataFrame."""
        if not issues:
            return pd.DataFrame(columns=[
                "id", "number", "state", "title", "user_login", "labels", 
                "assignees", "is_locked", "comments", "created_at", "updated_at",
                "closed_at", "url", "fetched_at"
            ])
        
        records = []
        for issue in issues:
            records.append({
                "id": issue.get("id", 0),
                "number": issue.get("number", 0),
                "state": issue.get("state", ""),
                "title": issue.get("title", "")[:200],
                "user_login": (issue.get("user") or {}).get("login", ""),
                "labels": ",".join([l.get("name", "") for l in issue.get("labels", [])]),
                "assignees": ",".join([a.get("login", "") for a in issue.get("assignees", [])]),
                "is_locked": issue.get("locked", False),
                "comments": issue.get("comments", 0),
                "created_at": issue.get("created_at", ""),
                "updated_at": issue.get("updated_at", ""),
                "closed_at": issue.get("closed_at", ""),
                "url": issue.get("html_url", ""),
                "fetched_at": datetime.utcnow().isoformat()
            })
        
        return pd.DataFrame(records)
    
    def _transform_pull_requests(self, prs: List[Dict]) -> pd.DataFrame:
        """Transform raw pull request data to DataFrame."""
        if not prs:
            return pd.DataFrame(columns=[
                "id", "number", "state", "title", "user_login", "draft",
                "created_at", "updated_at", "closed_at", "merged_at",
                "additions", "deletions", "changed_files", "comments",
                "review_comments", "url", "fetched_at"
            ])
        
        records = []
        for pr in prs:
            records.append({
                "id": pr.get("id", 0),
                "number": pr.get("number", 0),
                "state": pr.get("state", ""),
                "title": pr.get("title", "")[:200],
                "user_login": (pr.get("user") or {}).get("login", ""),
                "draft": pr.get("draft", False),
                "created_at": pr.get("created_at", ""),
                "updated_at": pr.get("updated_at", ""),
                "closed_at": pr.get("closed_at", ""),
                "merged_at": pr.get("merged_at", ""),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
                "comments": pr.get("comments", 0),
                "review_comments": pr.get("review_comments", 0),
                "url": pr.get("html_url", ""),
                "fetched_at": datetime.utcnow().isoformat()
            })
        
        return pd.DataFrame(records)
    
    def _transform_stargazers(self, stargazers: List[Dict]) -> pd.DataFrame:
        """Transform raw stargazer data to DataFrame."""
        if not stargazers:
            return pd.DataFrame(columns=[
                "user_login", "user_type", "starred_at", "site_admin", "fetched_at"
            ])
        
        records = []
        for star in stargazers:
            user = star.get("user", {}) or {}
            records.append({
                "user_login": user.get("login", ""),
                "user_type": user.get("type", ""),
                "starred_at": star.get("starred_at", ""),
                "site_admin": user.get("site_admin", False),
                "fetched_at": datetime.utcnow().isoformat()
            })
        
        return pd.DataFrame(records)
    
    def fetch_repository(
        self, 
        owner: str, 
        repo: str,
        force_update: bool = False,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch all data for a single repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            force_update: Force update even if recently fetched.
            incremental: Only fetch new data since last fetch.
            
        Returns:
            Dictionary with fetch results and statistics.
        """
        full_name = f"{owner}/{repo}"
        folder_name = f"{owner}__{repo}"
        repo_dir = self.data_dir / folder_name
        
        logger.info(f"Processing repository: {full_name}")
        
        # Check if update is needed
        if not force_update and self.tracker.is_repository_tracked(full_name):
            if not self.tracker.should_update(full_name):
                logger.info(f"Repository {full_name} recently fetched. Skipping.")
                return {
                    "status": "skipped",
                    "reason": "recently_fetched",
                    "full_name": full_name
                }
        
        # Determine incremental fetch parameters
        since = None
        if incremental and self.tracker.is_repository_tracked(full_name):
            since = self.tracker.get_last_fetch_time(full_name)
            logger.info(f"Incremental fetch since: {since}")
        
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Fetch data
            logger.info(f"Fetching commits for {full_name}...")
            commits = self.client.get_commits(owner, repo, since=since)
            commits_df = self._transform_commits(commits)
            
            logger.info(f"Fetching issues for {full_name}...")
            issues = self.client.get_issues(owner, repo, since=since)
            issues_df = self._transform_issues(issues)
            
            logger.info(f"Fetching pull requests for {full_name}...")
            prs = self.client.get_pull_requests(owner, repo)
            prs_df = self._transform_pull_requests(prs)
            
            logger.info(f"Fetching stargazers for {full_name}...")
            stargazers = self.client.get_stargazers(owner, repo)
            stargazers_df = self._transform_stargazers(stargazers)
            
            # Handle incremental updates
            if incremental and self.tracker.is_repository_tracked(full_name):
                commits_df = self._merge_incremental_data(
                    repo_dir / "commits.csv", commits_df, "sha"
                )
                issues_df = self._merge_incremental_data(
                    repo_dir / "issues.csv", issues_df, "id"
                )
                prs_df = self._merge_incremental_data(
                    repo_dir / "pull_requests.csv", prs_df, "id"
                )
                # Stargazers need special handling - replace entirely
            
            # Save data
            commits_df.to_csv(repo_dir / "commits.csv", index=False)
            issues_df.to_csv(repo_dir / "issues.csv", index=False)
            prs_df.to_csv(repo_dir / "pull_requests.csv", index=False)
            stargazers_df.to_csv(repo_dir / "stargazers.csv", index=False)
            
            # Compute hashes and update tracking
            data_hashes = {
                "commits": self._compute_data_hash(commits),
                "issues": self._compute_data_hash(issues),
                "pull_requests": self._compute_data_hash(prs),
                "stargazers": self._compute_data_hash(stargazers)
            }
            
            record_counts = {
                "commits": len(commits_df),
                "issues": len(issues_df),
                "pull_requests": len(prs_df),
                "stargazers": len(stargazers_df)
            }
            
            self.tracker.update_repository_tracking(full_name, data_hashes, record_counts)
            
            logger.info(f"Successfully fetched {full_name}: {record_counts}")
            
            return {
                "status": "success",
                "full_name": full_name,
                "folder": str(repo_dir),
                "record_counts": record_counts
            }
            
        except Exception as e:
            logger.error(f"Error fetching {full_name}: {e}")
            return {
                "status": "error",
                "full_name": full_name,
                "error": str(e)
            }
    
    def _merge_incremental_data(
        self, 
        existing_file: Path, 
        new_df: pd.DataFrame,
        unique_key: str
    ) -> pd.DataFrame:
        """Merge new data with existing data, avoiding duplicates."""
        if not existing_file.exists():
            return new_df
        
        try:
            existing_df = pd.read_csv(existing_file)
            
            if len(new_df) == 0:
                return existing_df
            
            # Combine and deduplicate
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[unique_key], keep="last")
            
            return combined
            
        except Exception as e:
            logger.warning(f"Error merging incremental data: {e}")
            return new_df
    
    def fetch_multiple_repositories(
        self, 
        repositories: List[Tuple[str, str]],
        max_workers: int = 3,
        force_update: bool = False
    ) -> List[Dict]:
        """
        Fetch data for multiple repositories concurrently.
        
        Args:
            repositories: List of (owner, repo) tuples.
            max_workers: Maximum concurrent workers.
            force_update: Force update for all repositories.
            
        Returns:
            List of fetch results.
        """
        results = []
        
        # Use sequential processing to respect rate limits
        for owner, repo in repositories:
            result = self.fetch_repository(owner, repo, force_update=force_update)
            results.append(result)
            
            # Small delay between repositories to be nice to GitHub API
            time.sleep(0.5)
        
        return results
    
    def fetch_from_summary_csv(
        self, 
        summary_file: str,
        limit: Optional[int] = None,
        force_update: bool = False
    ) -> List[Dict]:
        """
        Fetch repositories listed in a summary CSV file.
        
        Args:
            summary_file: Path to the repositories_summary.csv file.
            limit: Maximum number of repositories to fetch.
            force_update: Force update for all repositories.
            
        Returns:
            List of fetch results.
        """
        summary_df = pd.read_csv(summary_file)
        
        repositories = []
        for _, row in summary_df.iterrows():
            full_name = row.get("full_name", "")
            if "/" in full_name:
                owner, repo = full_name.split("/", 1)
                repositories.append((owner, repo))
        
        if limit:
            repositories = repositories[:limit]
        
        logger.info(f"Fetching {len(repositories)} repositories from summary file")
        
        return self.fetch_multiple_repositories(
            repositories, 
            force_update=force_update
        )


def create_ingestion_pipeline(
    token: Optional[str] = None,
    data_dir: str = "repositories"
) -> DataIngestionPipeline:
    """
    Factory function to create a data ingestion pipeline.
    
    Args:
        token: GitHub token. If not provided, uses GITHUB_TOKEN env var.
        data_dir: Directory to store repository data.
        
    Returns:
        Configured DataIngestionPipeline instance.
    """
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GitHub token required. Provide via argument or GITHUB_TOKEN env var."
            )
    
    return DataIngestionPipeline(github_token=token, data_dir=data_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Data Ingestion Pipeline")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument("--repo", help="Single repository to fetch (owner/repo format)")
    parser.add_argument("--summary", help="Path to repositories_summary.csv")
    parser.add_argument("--limit", type=int, help="Limit number of repos to fetch")
    parser.add_argument("--force", action="store_true", help="Force update")
    parser.add_argument("--data-dir", default="repositories", help="Data directory")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub token required via --token or GITHUB_TOKEN env var")
        exit(1)
    
    pipeline = DataIngestionPipeline(
        github_token=token,
        data_dir=args.data_dir
    )
    
    # Verify token
    if not pipeline.client.check_token_validity():
        print("Error: Invalid GitHub token")
        exit(1)
    
    if args.repo:
        owner, repo = args.repo.split("/")
        result = pipeline.fetch_repository(owner, repo, force_update=args.force)
        print(f"Result: {result}")
    
    elif args.summary:
        results = pipeline.fetch_from_summary_csv(
            args.summary, 
            limit=args.limit,
            force_update=args.force
        )
        
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")
        
        print(f"\nFetch complete: {success} success, {skipped} skipped, {errors} errors")
    
    else:
        print("Specify --repo or --summary to fetch data")
