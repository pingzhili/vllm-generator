import pytest
import json

from src import ModelConfig, GenerationConfig


class TestModelConfig:
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ModelConfig(model="gpt2")
        
        assert config.model == "gpt2"
        assert config.dtype == "auto"
        assert config.temperature == 1.0
        assert config.max_tokens == 512
        assert config.gpu_memory_utilization == 0.9
        assert config.tensor_parallel_size == 1
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = ModelConfig(model="gpt2")
        config.validate()  # Should not raise
        
        # Invalid temperature
        config = ModelConfig(model="gpt2", temperature=-1)
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            config.validate()
        
        # Invalid top_p
        config = ModelConfig(model="gpt2", top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            config.validate()
        
        # Invalid gpu_memory_utilization
        config = ModelConfig(model="gpt2", gpu_memory_utilization=1.5)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0 and 1"):
            config.validate()
    
    def test_to_vllm_args(self):
        """Test conversion to vLLM arguments"""
        config = ModelConfig(
            model="meta-llama/Llama-2-7b-hf",
            dtype="float16",
            gpu_memory_utilization=0.95,
            tensor_parallel_size=2,
            max_model_len=2048,
            quantization="awq"
        )
        
        vllm_args = config.to_vllm_args()
        
        assert vllm_args["model"] == "meta-llama/Llama-2-7b-hf"
        assert vllm_args["dtype"] == "float16"
        assert vllm_args["gpu_memory_utilization"] == 0.95
        assert vllm_args["tensor_parallel_size"] == 2
        assert vllm_args["max_model_len"] == 2048
        assert vllm_args["quantization"] == "awq"
    
    def test_to_sampling_params(self):
        """Test conversion to sampling parameters"""
        config = ModelConfig(
            model="gpt2",
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            max_tokens=256,
            repetition_penalty=1.1,
            stop_sequences=["\\n", "END"],
            seed=42
        )
        
        sampling_params = config.to_sampling_params()
        
        assert sampling_params["temperature"] == 0.8
        assert sampling_params["top_p"] == 0.95
        assert sampling_params["top_k"] == 50
        assert sampling_params["max_tokens"] == 256
        assert sampling_params["repetition_penalty"] == 1.1
        assert sampling_params["stop"] == ["\\n", "END"]
        assert sampling_params["seed"] == 42
    
    def test_from_yaml(self, temp_dir):
        """Test loading config from YAML"""
        yaml_content = """
model: meta-llama/Llama-2-7b-hf
temperature: 0.7
max_tokens: 1024
gpu_memory_utilization: 0.95
"""
        yaml_file = temp_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        config = ModelConfig.from_yaml(str(yaml_file))
        
        assert config.model == "meta-llama/Llama-2-7b-hf"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.gpu_memory_utilization == 0.95
    
    def test_from_json(self, temp_dir):
        """Test loading config from JSON"""
        json_content = {
            "model": "gpt2",
            "temperature": 0.5,
            "max_tokens": 512
        }
        json_file = temp_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(json_content, f)
        
        config = ModelConfig.from_json(str(json_file))
        
        assert config.model == "gpt2"
        assert config.temperature == 0.5
        assert config.max_tokens == 512
    
    def test_update_config(self):
        """Test updating configuration"""
        config = ModelConfig(model="gpt2")
        
        config.update(temperature=0.8, max_tokens=256)
        
        assert config.temperature == 0.8
        assert config.max_tokens == 256
        
        # Try invalid field
        with pytest.raises(ValueError, match="Unknown config parameter"):
            config.update(invalid_field=123)


class TestGenerationConfig:
    
    def test_default_generation_config(self):
        """Test default generation configuration"""
        config = GenerationConfig()
        
        assert config.batch_size == 32
        assert config.num_repeats == 1
        assert config.repeat_strategy == "independent"
        assert config.checkpoint_frequency == 100
        assert config.error_handling == "skip"
        assert config.max_retries == 3
    
    def test_generation_config_from_dict(self):
        """Test creating generation config from dictionary"""
        config_dict = {
            "batch_size": 64,
            "num_repeats": 5,
            "repeat_strategy": "temperature_schedule",
            "temperature_schedule": [0.5, 0.7, 0.9, 1.1, 1.3],
            "checkpoint_frequency": 50
        }
        
        config = GenerationConfig.from_dict(config_dict)
        
        assert config.batch_size == 64
        assert config.num_repeats == 5
        assert config.repeat_strategy == "temperature_schedule"
        assert config.temperature_schedule == [0.5, 0.7, 0.9, 1.1, 1.3]
        assert config.checkpoint_frequency == 50