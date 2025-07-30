"""Helper utilities for the vLLM generator."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datetime import datetime


def generate_batch_id(data: Union[str, Dict[str, Any]]) -> str:
    """Generate a unique batch ID from data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()[:8]


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """Split dataframe into chunks."""
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def distribute_items(items: List[Any], n_workers: int) -> List[List[Any]]:
    """Distribute items evenly among workers."""
    if n_workers <= 0:
        raise ValueError("Number of workers must be positive")
    
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}
    
    import statistics
    
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }