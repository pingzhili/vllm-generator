"""Tests for configuration parser."""

import pytest
from pathlib import Path
import yaml

from vllm_generator.config import ConfigParser, Config
from vllm_generator.config.schemas import (
    DataConfig, ModelConfig, GenerationConfig,
    LoggingConfig
)


class TestConfigParser:
    """Test configuration parser functionality."""
    
    def test_from_dict(self, sample_config_dict):
        """Test creating config from dictionary."""
        config = ConfigParser.from_dict(sample_config_dict)
        
        assert isinstance(config, Config)
        assert config.data.input_path == Path("test_input.parquet")
        assert config.data.output_path == Path("test_output.parquet")
        assert len(config.models) == 1
        assert str(config.models[0].url) == "http://localhost:8000/"
        assert config.generation.num_samples == 1
        assert config.processing.batch_size == 2
    
    def test_from_yaml(self, temp_dir, sample_config_dict):
        """Test loading config from YAML file."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)
        
        config = ConfigParser.from_yaml(config_file)
        
        assert isinstance(config, Config)
        assert config.data.input_column == "question"
        assert config.generation.temperature == 1.0
    
    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ConfigParser.from_yaml(Path("nonexistent.yaml"))
    
    def test_merge_cli_args(self, sample_config_dict):
        """Test merging CLI arguments into config."""
        config = ConfigParser.from_dict(sample_config_dict)
        
        cli_args = {
            "input": Path("new_input.parquet"),
            "output": Path("new_output.parquet"),
            "num_samples": 5,
            "temperature": 0.8,
            "batch_size": 64,
            "model_url": "http://new-server:8000"
        }
        
        merged_config = ConfigParser.merge_cli_args(config, cli_args)
        
        assert merged_config.data.input_path == Path("new_input.parquet")
        assert merged_config.data.output_path == Path("new_output.parquet")
        assert merged_config.generation.num_samples == 5
        assert merged_config.generation.temperature == 0.8
        assert merged_config.processing.batch_size == 64
        assert str(merged_config.models[0].url) == "http://new-server:8000/"
    
    def test_merge_cli_args_model_urls(self, sample_config_dict):
        """Test merging multiple model URLs from CLI."""
        config = ConfigParser.from_dict(sample_config_dict)
        
        cli_args = {
            "model_urls": ["http://server1:8000", "http://server2:8000"]
        }
        
        merged_config = ConfigParser.merge_cli_args(config, cli_args)
        
        assert len(merged_config.models) == 2
        assert str(merged_config.models[0].url) == "http://server1:8000/"
        assert str(merged_config.models[1].url) == "http://server2:8000/"
    
    def test_to_yaml(self, temp_dir, sample_config_dict):
        """Test saving config to YAML file."""
        config = ConfigParser.from_dict(sample_config_dict)
        
        output_file = temp_dir / "output_config.yaml"
        ConfigParser.to_yaml(config, output_file)
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["data"]["input_column"] == "question"
        assert loaded_data["generation"]["num_samples"] == 1
    
    def test_create_default_config(self):
        """Test creating default config with minimal parameters."""
        config = ConfigParser.create_default_config(
            input_path="input.parquet",
            output_path="output.parquet",
            model_url="http://localhost:8000",
            num_samples=3,
            temperature=0.7
        )
        
        assert config.data.input_path == Path("input.parquet")
        assert config.data.output_path == Path("output.parquet")
        assert str(config.models[0].url) == "http://localhost:8000/"
        assert config.generation.num_samples == 3
        assert config.generation.temperature == 0.7


class TestConfigSchemas:
    """Test configuration schema validation."""
    
    def test_data_config_validation(self):
        """Test DataConfig validation."""
        # Valid config
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=Path("output.parquet")
        )
        assert config.input_column == "question"  # default
        assert config.output_column == "response"  # default
        
        # Invalid file extension
        with pytest.raises(ValueError, match="Input file must be a parquet file"):
            DataConfig(
                input_path=Path("input.csv"),
                output_path=Path("output.parquet")
            )
        
        with pytest.raises(ValueError, match="Output file must be a parquet file"):
            DataConfig(
                input_path=Path("input.parquet"),
                output_path=Path("output.json")
            )
    
    def test_generation_config_temperature_validation(self):
        """Test temperature validation in GenerationConfig."""
        # Single temperature
        config = GenerationConfig(temperature=0.8)
        assert config.temperature == 0.8
        
        # Temperature list matching num_samples
        config = GenerationConfig(
            num_samples=3,
            temperature=[0.5, 0.7, 0.9]
        )
        assert config.temperature == [0.5, 0.7, 0.9]
        
        # Temperature list not matching num_samples
        with pytest.raises(ValueError, match="Temperature list length"):
            GenerationConfig(
                num_samples=3,
                temperature=[0.5, 0.7]
            )
    
    def test_logging_config_validation(self):
        """Test LoggingConfig validation."""
        # Valid log level
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"  # Should be uppercase
        
        # Invalid log level
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfig(level="INVALID")
    
    def test_config_validation(self):
        """Test main Config validation."""
        # No models
        with pytest.raises(ValueError, match="At least one model must be configured"):
            Config(
                data=DataConfig(
                    input_path=Path("input.parquet"),
                    output_path=Path("output.parquet")
                ),
                models=[]
            )
        
        # Valid config
        config = Config(
            data=DataConfig(
                input_path=Path("input.parquet"),
                output_path=Path("output.parquet")
            ),
            models=[ModelConfig(url="http://localhost:8000")]
        )
        assert len(config.models) == 1