import pytest
import time

from src.models.config import ModelConfig
from src.models.vllm_wrapper import MockVLLMModel, VLLMModel


class TestMockVLLMModel:
    """Test the mock vLLM model (which works on Mac)"""
    
    def test_mock_model_initialization(self):
        """Test mock model initialization"""
        config = ModelConfig(model="gpt2")
        model = MockVLLMModel(config)
        
        assert model.config == config
        assert len(model.response_templates) > 0
    
    def test_mock_model_generate(self):
        """Test mock model generation"""
        config = ModelConfig(model="gpt2", temperature=0.7)
        model = MockVLLMModel(config)
        
        prompts = [
            "What is AI?",
            "Explain quantum computing",
            "How does the internet work?"
        ]
        
        results = model.generate(prompts)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["prompt"] == prompts[i]
            assert isinstance(result["response"], str)
            assert result["tokens"] > 0
            assert result["latency"] > 0
            assert result["finish_reason"] == "stop"
    
    def test_mock_model_temperature_variation(self):
        """Test that mock model responds to temperature changes"""
        config = ModelConfig(model="gpt2")
        model = MockVLLMModel(config)
        
        # Low temperature
        results_low = model.generate(
            ["Test prompt"],
            {"temperature": 0.1}
        )
        
        # High temperature
        results_high = model.generate(
            ["Test prompt"],
            {"temperature": 1.5}
        )
        
        # High temperature should have additional text
        assert "Temperature:" in results_high[0]["response"]
        assert "Temperature:" not in results_low[0]["response"]
    
    def test_mock_model_streaming(self):
        """Test mock model streaming generation"""
        config = ModelConfig(model="gpt2")
        model = MockVLLMModel(config)
        
        prompts = ["Generate a story"]
        stream_results = list(model.generate_stream(prompts))
        
        assert len(stream_results) > 0
        
        # Check streaming format
        for result in stream_results[:-1]:
            assert result["prompt_idx"] == 0
            assert isinstance(result["text"], str)
            assert result["finished"] is False
        
        # Last result should be finished
        assert stream_results[-1]["finished"] is True
    
    def test_mock_model_shutdown(self):
        """Test mock model shutdown"""
        config = ModelConfig(model="gpt2")
        model = MockVLLMModel(config)
        
        # Should not raise any errors
        model.shutdown()


class TestVLLMModel:
    """Test the VLLMModel wrapper"""
    
    def test_vllm_model_fallback_to_mock(self):
        """Test that VLLMModel falls back to MockVLLMModel on Mac"""
        config = ModelConfig(model="gpt2")
        model = VLLMModel(config)
        
        # On Mac, should fall back to MockVLLMModel
        assert isinstance(model, MockVLLMModel)
        
        # Should still work normally
        results = model.generate(["Test prompt"])
        assert len(results) == 1
        assert isinstance(results[0]["response"], str)
    
    def test_model_config_to_dict(self):
        """Test model configuration serialization"""
        config = ModelConfig(
            model="meta-llama/Llama-2-7b-hf",
            temperature=0.8,
            max_tokens=256,
            stop_sequences=["\\n", "END"]
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model"] == "meta-llama/Llama-2-7b-hf"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 256
        assert config_dict["stop_sequences"] == ["\\n", "END"]