"""Tests for enable_thinking parameter and extra_body handling."""

import pytest
from vllm_generator.config.schemas import GenerationConfig
from vllm_generator.models import VLLMClient
from vllm_generator.config.schemas import ModelConfig, RetryConfig


class TestEnableThinking:
    """Test enable_thinking functionality."""
    
    def test_generation_config_enable_thinking(self):
        """Test GenerationConfig with enable_thinking."""
        # Default should be False
        config = GenerationConfig()
        assert config.enable_thinking is False
        
        # Explicit True
        config = GenerationConfig(enable_thinking=True)
        assert config.enable_thinking is True
        
        # With extra_body
        config = GenerationConfig(
            enable_thinking=True,
            extra_body={"custom": "value"}
        )
        assert config.enable_thinking is True
        assert config.extra_body == {"custom": "value"}
    
    def test_vllm_client_prepare_request_with_thinking(self):
        """Test VLLMClient request preparation with enable_thinking."""
        model_config = ModelConfig(url="http://localhost:8000")
        generation_config = GenerationConfig(
            enable_thinking=True,
            max_tokens=1024,
            temperature=0.7,
            top_k=20
        )
        retry_config = RetryConfig()
        
        client = VLLMClient(model_config, generation_config, retry_config)
        request = client._prepare_request("Test prompt")
        
        # Check basic parameters
        assert request["prompt"] == "Test prompt"
        assert request["max_tokens"] == 1024
        assert request["temperature"] == 0.7
        
        # Check extra_body
        assert "extra_body" in request
        assert "chat_template_kwargs" in request["extra_body"]
        assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert request["extra_body"]["top_k"] == 20
    
    def test_vllm_client_prepare_request_without_thinking(self):
        """Test VLLMClient request preparation without enable_thinking."""
        model_config = ModelConfig(url="http://localhost:8000")
        generation_config = GenerationConfig(
            enable_thinking=False,
            max_tokens=512
        )
        retry_config = RetryConfig()
        
        client = VLLMClient(model_config, generation_config, retry_config)
        request = client._prepare_request("Test prompt")
        
        # Should not have extra_body if enable_thinking is False and top_k is -1
        assert "extra_body" not in request or request["extra_body"] == {}
    
    def test_vllm_client_merge_extra_body(self):
        """Test merging custom extra_body with enable_thinking."""
        model_config = ModelConfig(url="http://localhost:8000")
        generation_config = GenerationConfig(
            enable_thinking=True,
            top_k=50,
            extra_body={
                "custom_param": "value",
                "chat_template_kwargs": {
                    "custom_mode": "fast"
                }
            }
        )
        retry_config = RetryConfig()
        
        client = VLLMClient(model_config, generation_config, retry_config)
        request = client._prepare_request("Test prompt")
        
        # Check merged extra_body
        assert "extra_body" in request
        assert request["extra_body"]["custom_param"] == "value"
        assert request["extra_body"]["top_k"] == 50
        
        # Check merged chat_template_kwargs
        chat_kwargs = request["extra_body"]["chat_template_kwargs"]
        assert chat_kwargs["enable_thinking"] is True
        assert chat_kwargs["custom_mode"] == "fast"
    
    def test_config_parser_cli_enable_thinking(self):
        """Test parsing enable_thinking from CLI args."""
        from vllm_generator.config.parser import ConfigParser
        
        base_config = ConfigParser.create_default_config(
            input_path="input.parquet",
            output_path="output.parquet",
            model_url="http://localhost:8000"
        )
        
        # Merge CLI args with enable_thinking
        cli_args = {"enable_thinking": True}
        merged_config = ConfigParser.merge_cli_args(base_config, cli_args)
        
        assert merged_config.generation.enable_thinking is True
    
    def test_yaml_config_with_enable_thinking(self, temp_dir):
        """Test YAML configuration with enable_thinking."""
        from vllm_generator.config.parser import ConfigParser
        import yaml
        
        config_data = {
            "data": {
                "input_path": "input.parquet",
                "output_path": "output.parquet"
            },
            "models": [{"url": "http://localhost:8000"}],
            "generation": {
                "enable_thinking": True,
                "max_tokens": 8192,
                "extra_body": {
                    "custom": "value"
                }
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = ConfigParser.from_yaml(config_file)
        
        assert config.generation.enable_thinking is True
        assert config.generation.max_tokens == 8192
        assert config.generation.extra_body == {"custom": "value"}