import importlib.util
from typing import Dict, Any, List
from pathlib import Path
import re


def parse_args_string(args_string: str) -> Dict[str, Any]:
    """Parse key=value argument string into dictionary
    
    Examples:
        "model=gpt2,temperature=0.8" -> {"model": "gpt2", "temperature": 0.8}
    """
    if not args_string:
        return {}
    
    args_dict = {}
    
    # Split by comma, but handle quoted strings and brackets
    parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', args_string)
    
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Remove quotes if present
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        # Try to parse value type
        parsed_value = parse_value(value)
        args_dict[key] = parsed_value
    
    return args_dict


def parse_value(value: str) -> Any:
    """Parse string value to appropriate type"""
    # Boolean
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    
    # None
    if value.lower() == "none":
        return None
    
    # Number
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # List (comma-separated in brackets)
    if value.startswith('[') and value.endswith(']'):
        list_content = value[1:-1]
        if list_content:
            items = [item.strip() for item in list_content.split(',')]
            return [parse_value(item) for item in items]
        return []
    
    # Default to string
    return value


def load_module_from_path(module_path: str, module_name: str = "custom_module"):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gpu_list(gpu_string: str) -> List[int]:
    """Parse GPU list from string
    
    Examples:
        "0,1,2,3" -> [0, 1, 2, 3]
        "0-3" -> [0, 1, 2, 3]
        "0,2-4,6" -> [0, 2, 3, 4, 6]
    """
    if not gpu_string:
        return []
    
    gpus = []
    parts = gpu_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range
            start, end = part.split('-')
            gpus.extend(range(int(start), int(end) + 1))
        else:
            # Single GPU
            gpus.append(int(part))
    
    return sorted(list(set(gpus)))  # Remove duplicates and sort


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_output_path(output_path: str, overwrite: bool = False) -> Path:
    """Validate and prepare output path"""
    path = Path(output_path)
    
    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. "
            "Use --overwrite to overwrite."
        )
    
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged