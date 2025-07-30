import json
import yaml
from pathlib import Path
from typing import Dict, Any
import os

from ..utils.helpers import merge_configs


class ConfigParser:
    """Parse configuration from files and environment variables"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Determine file type
        if path.suffix in ['.yaml', '.yml']:
            return ConfigParser._load_yaml(path)
        elif path.suffix == '.json':
            return ConfigParser._load_json(path)
        else:
            raise ValueError(f"Unsupported config file type: {path.suffix}")
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables
        
        Environment variables should be prefixed with VLLM_GEN_
        """
        config = {}
        prefix = "VLLM_GEN_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Convert underscores to dots for nested keys
                # e.g., VLLM_GEN_MODEL_CONFIG_TEMPERATURE -> model_config.temperature
                if '_' in config_key:
                    parts = config_key.split('_')
                    
                    # Build nested dict
                    current = config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Set value
                    current[parts[-1]] = ConfigParser._parse_env_value(value)
                else:
                    config[config_key] = ConfigParser._parse_env_value(value)
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except:
            # Fall back to string
            return value
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations
        
        Later configs override earlier ones
        """
        result = {}
        
        for config in configs:
            if config:
                result = merge_configs(result, config)
        
        return result
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """Save configuration to file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
    
    @staticmethod
    def create_example_config() -> Dict[str, Any]:
        """Create an example configuration"""
        return {
            "model_config": {
                "model": "meta-llama/Llama-2-7b-hf",
                "dtype": "float16",
                "temperature": 0.7,
                "max_tokens": 512,
                "gpu_memory_utilization": 0.9
            },
            "generation_config": {
                "batch_size": 32,
                "num_repeats": 1,
                "error_handling": "skip",
                "checkpoint_frequency": 100
            },
            "data_config": {
                "question_column": "question",
                "output_format": "wide",
                "output_column_prefix": "response"
            },
            "parallel_config": {
                "mode": "single",
                "num_workers": 1
            },
            "logging_config": {
                "level": "INFO",
                "progress_bar": True
            }
        }