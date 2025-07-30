"""Configuration parser for YAML files and CLI arguments."""

import yaml
from pathlib import Path
from typing import Dict, Any
from .schemas import Config


class ConfigParser:
    """Parse and validate configuration from YAML files and CLI arguments."""
    
    @staticmethod
    def from_yaml(config_path: Path) -> Config:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        return Config(**data)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        return Config(**data)
    
    @staticmethod
    def merge_cli_args(config: Config, args: Dict[str, Any]) -> Config:
        """Merge CLI arguments into existing configuration."""
        # Create a copy of the configuration data
        config_data = config.model_dump()
        
        # Map CLI arguments to configuration structure
        if args.get("input"):
            config_data["data"]["input_path"] = args["input"]
        if args.get("output"):
            config_data["data"]["output_path"] = args["output"]
        if args.get("input_column"):
            config_data["data"]["input_column"] = args["input_column"]
        if args.get("output_column"):
            config_data["data"]["output_column"] = args["output_column"]
        
        # Model configuration
        if args.get("model_url"):
            config_data["models"] = [{"url": args["model_url"]}]
        elif args.get("model_urls"):
            config_data["models"] = [{"url": url} for url in args["model_urls"]]
        
        # Generation parameters
        if args.get("num_samples") is not None:
            config_data["generation"]["num_samples"] = args["num_samples"]
        if args.get("temperature") is not None:
            config_data["generation"]["temperature"] = args["temperature"]
        if args.get("top_p") is not None:
            config_data["generation"]["top_p"] = args["top_p"]
        if args.get("top_k") is not None:
            config_data["generation"]["top_k"] = args["top_k"]
        if args.get("max_tokens") is not None:
            config_data["generation"]["max_tokens"] = args["max_tokens"]
        if args.get("stop_sequences"):
            config_data["generation"]["stop_sequences"] = args["stop_sequences"]
        if args.get("seed") is not None:
            config_data["generation"]["seed"] = args["seed"]
        
        # Processing parameters
        if args.get("batch_size") is not None:
            config_data["processing"]["batch_size"] = args["batch_size"]
        if args.get("num_workers") is not None:
            config_data["processing"]["num_workers"] = args["num_workers"]
        if args.get("checkpoint_dir"):
            config_data["processing"]["checkpoint_dir"] = args["checkpoint_dir"]
        if args.get("resume"):
            config_data["processing"]["resume"] = args["resume"]
        
        # Retry parameters
        if args.get("timeout") is not None:
            config_data["retry"]["timeout"] = args["timeout"]
        if args.get("max_retries") is not None:
            config_data["retry"]["max_retries"] = args["max_retries"]
        if args.get("retry_delay") is not None:
            config_data["retry"]["retry_delay"] = args["retry_delay"]
        
        return Config(**config_data)
    
    @staticmethod
    def to_yaml(config: Config, output_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = config.model_dump(mode="json")
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def create_default_config(
        input_path: str,
        output_path: str,
        model_url: str,
        **kwargs
    ) -> Config:
        """Create a default configuration with minimal required parameters."""
        config_data = {
            "data": {
                "input_path": input_path,
                "output_path": output_path,
                "input_column": kwargs.get("input_column", "question"),
                "output_column": kwargs.get("output_column", "response"),
            },
            "models": [{"url": model_url}],
            "generation": {
                "num_samples": kwargs.get("num_samples", 1),
                "temperature": kwargs.get("temperature", 1.0),
                "max_tokens": kwargs.get("max_tokens", 512),
            },
            "processing": {
                "batch_size": kwargs.get("batch_size", 32),
                "num_workers": kwargs.get("num_workers", 1),
            }
        }
        
        return Config(**config_data)