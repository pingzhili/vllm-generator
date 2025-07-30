import pytest
import os
import json

from src import ConfigParser
from src import validate_config, create_config_from_args


class TestConfigParser:
    
    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML configuration"""
        yaml_content = """
model_config:
  model: meta-llama/Llama-2-7b-hf
  temperature: 0.7
  max_tokens: 512
generation_config:
  batch_size: 32
  num_repeats: 3
"""
        yaml_file = temp_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        config = ConfigParser.load_config(str(yaml_file))
        
        assert config["model_config"]["model"] == "meta-llama/Llama-2-7b-hf"
        assert config["model_config"]["temperature"] == 0.7
        assert config["generation_config"]["batch_size"] == 32
    
    def test_load_json_config(self, temp_dir):
        """Test loading JSON configuration"""
        json_content = {
            "model_config": {
                "model": "gpt2",
                "temperature": 0.5
            },
            "generation_config": {
                "batch_size": 64
            }
        }
        json_file = temp_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(json_content, f)
        
        config = ConfigParser.load_config(str(json_file))
        
        assert config["model_config"]["model"] == "gpt2"
        assert config["model_config"]["temperature"] == 0.5
        assert config["generation_config"]["batch_size"] == 64
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables"""
        # Set environment variables
        os.environ["VLLM_GEN_MODEL_CONFIG_TEMPERATURE"] = "0.8"
        os.environ["VLLM_GEN_GENERATION_CONFIG_BATCH_SIZE"] = "128"
        os.environ["VLLM_GEN_PARALLEL_CONFIG_NUM_WORKERS"] = "4"
        
        try:
            config = ConfigParser.load_from_env()
            
            assert config["model"]["config"]["temperature"] == 0.8
            assert config["generation"]["config"]["batch"]["size"] == 128
            assert config["parallel"]["config"]["num"]["workers"] == 4
        finally:
            # Clean up
            for key in list(os.environ.keys()):
                if key.startswith("VLLM_GEN_"):
                    del os.environ[key]
    
    def test_merge_configs(self):
        """Test merging multiple configurations"""
        base_config = {
            "model_config": {
                "model": "gpt2",
                "temperature": 0.7,
                "max_tokens": 512
            },
            "generation_config": {
                "batch_size": 32
            }
        }
        
        override_config = {
            "model_config": {
                "temperature": 0.9,
                "top_p": 0.95
            },
            "generation_config": {
                "batch_size": 64,
                "num_repeats": 3
            }
        }
        
        merged = ConfigParser.merge_configs(base_config, override_config)
        
        assert merged["model_config"]["model"] == "gpt2"  # Preserved
        assert merged["model_config"]["temperature"] == 0.9  # Overridden
        assert merged["model_config"]["max_tokens"] == 512  # Preserved
        assert merged["model_config"]["top_p"] == 0.95  # Added
        assert merged["generation_config"]["batch_size"] == 64  # Overridden
        assert merged["generation_config"]["num_repeats"] == 3  # Added
    
    def test_save_config(self, temp_dir):
        """Test saving configuration"""
        config = {
            "model_config": {
                "model": "gpt2",
                "temperature": 0.7
            }
        }
        
        # Save as YAML
        yaml_path = temp_dir / "saved_config.yaml"
        ConfigParser.save_config(config, str(yaml_path))
        
        assert yaml_path.exists()
        loaded_yaml = ConfigParser.load_config(str(yaml_path))
        assert loaded_yaml == config
        
        # Save as JSON
        json_path = temp_dir / "saved_config.json"
        ConfigParser.save_config(config, str(json_path))
        
        assert json_path.exists()
        loaded_json = ConfigParser.load_config(str(json_path))
        assert loaded_json == config
    
    def test_create_example_config(self):
        """Test creating example configuration"""
        config = ConfigParser.create_example_config()
        
        assert "model_config" in config
        assert "generation_config" in config
        assert "data_config" in config
        assert "parallel_config" in config
        assert "logging_config" in config
        
        assert config["model_config"]["model"] == "meta-llama/Llama-2-7b-hf"
        assert config["generation_config"]["batch_size"] == 32
        assert config["parallel_config"]["mode"] == "single"
    
    def test_invalid_config_file(self, temp_dir):
        """Test handling of invalid config file"""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            ConfigParser.load_config("non_existent.yaml")
        
        # Unsupported file type
        txt_file = temp_dir / "config.txt"
        txt_file.write_text("invalid config")
        
        with pytest.raises(ValueError, match="Unsupported config file type"):
            ConfigParser.load_config(str(txt_file))


class TestConfigValidation:
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration"""
        config = {
            "model_config": {
                "model": "gpt2",
                "temperature": 0.7,
                "top_p": 0.95,
                "gpu_memory_utilization": 0.9,
                "max_tokens": 512
            },
            "generation_config": {
                "batch_size": 32,
                "num_repeats": 3,
                "repeat_strategy": "independent"
            }
        }
        
        errors = validate_config(config)
        assert len(errors) == 0
    
    def test_validate_invalid_config(self):
        """Test validation of invalid configuration"""
        config = {
            "model_config": {
                # Missing required model field
                "temperature": -1,  # Invalid
                "top_p": 1.5,  # Invalid
                "gpu_memory_utilization": 1.5,  # Invalid
                "max_tokens": -10  # Invalid
            },
            "generation_config": {
                "batch_size": 0,  # Invalid
                "num_repeats": -1,  # Invalid
                "repeat_strategy": "invalid_strategy"  # Invalid
            }
        }
        
        errors = validate_config(config)
        
        assert "model" in errors
        assert "temperature" in errors
        assert "top_p" in errors
        assert "gpu_memory_utilization" in errors
        assert "max_tokens" in errors
        assert "batch_size" in errors
        assert "num_repeats" in errors
        assert "repeat_strategy" in errors
    
    def test_validate_temperature_schedule(self):
        """Test validation of temperature schedule"""
        # Valid temperature schedule
        config = {
            "model_config": {"model": "gpt2"},
            "generation_config": {
                "num_repeats": 3,
                "repeat_strategy": "temperature_schedule",
                "temperature_schedule": [0.5, 0.7, 0.9]
            }
        }
        errors = validate_config(config)
        assert len(errors) == 0
        
        # Missing temperature schedule
        config["generation_config"]["temperature_schedule"] = None
        errors = validate_config(config)
        assert "temperature_schedule" in errors
        
        # Wrong length
        config["generation_config"]["temperature_schedule"] = [0.5, 0.7]
        errors = validate_config(config)
        assert "temperature_schedule" in errors
    
    def test_create_config_from_args(self):
        """Test creating configuration from parsed arguments"""
        args = {
            "model": "gpt2",
            "temperature": 0.8,
            "max_tokens": 256,
            "batch_size": 64,
            "num_repeats": 3,
            "parallel_mode": "multi_server",
            "num_workers": 4,
            "log_level": "DEBUG",
            "output_dir": "/tmp/outputs"
        }
        
        config = create_config_from_args(args)
        
        assert config["model_config"]["model"] == "gpt2"
        assert config["model_config"]["temperature"] == 0.8
        assert config["generation_config"]["batch_size"] == 64
        assert config["parallel_config"]["parallel_mode"] == "multi_server"
        assert config["logging_config"]["log_level"] == "DEBUG"
        assert config["output_dir"] == "/tmp/outputs"