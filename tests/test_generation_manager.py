import pytest
import time
from unittest.mock import Mock, patch

from src.models.config import ModelConfig, GenerationConfig
from src.models.generation import GenerationManager
from src.models.vllm_wrapper import MockVLLMModel


class TestGenerationManager:
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        config = ModelConfig(model="gpt2")
        return MockVLLMModel(config)
    
    @pytest.fixture
    def generation_config(self):
        """Create generation configuration"""
        return GenerationConfig(
            batch_size=2,
            num_repeats=1,
            error_handling="skip",
            max_retries=2
        )
    
    def test_generation_manager_init(self, mock_model, generation_config):
        """Test generation manager initialization"""
        manager = GenerationManager(mock_model, generation_config)
        
        assert manager.model == mock_model
        assert manager.config == generation_config
        assert manager.track_metrics is True
        assert manager.metrics["total_prompts"] == 0
    
    def test_generate_batch(self, mock_model, generation_config):
        """Test batch generation"""
        manager = GenerationManager(mock_model, generation_config)
        
        items = [
            {"idx": 0, "prompt": "What is AI?"},
            {"idx": 1, "prompt": "Explain ML"},
            {"idx": 2, "prompt": "What is DL?"}
        ]
        
        results = manager.generate_batch(items)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["idx"] == i
            assert result["prompt"] == items[i]["prompt"]
            assert "response" in result
            assert "batch_idx" in result
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics["total_prompts"] == 3
        assert metrics["successful_generations"] == 3
    
    def test_generate_with_repeats(self, mock_model):
        """Test generation with multiple repeats"""
        config = GenerationConfig(
            batch_size=2,
            num_repeats=3,
            repeat_strategy="independent"
        )
        manager = GenerationManager(mock_model, config)
        
        items = [
            {"idx": 0, "prompt": "Test prompt"}
        ]
        
        results = manager.generate_with_repeats(
            items,
            num_repeats=3,
            repeat_strategy="independent"
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["repeat_id"] == i
            assert result["prompt"] == "Test prompt"
    
    def test_temperature_schedule(self, mock_model):
        """Test temperature schedule for repeats"""
        config = GenerationConfig()
        manager = GenerationManager(mock_model, config)
        
        items = [{"idx": 0, "prompt": "Test"}]
        temperature_schedule = [0.5, 1.0, 1.5]
        
        # Mock the generate_batch method to capture sampling params
        original_generate = manager.model.generate
        captured_params = []
        
        def mock_generate(prompts, sampling_params=None):
            captured_params.append(sampling_params)
            return original_generate(prompts, sampling_params)
        
        manager.model.generate = mock_generate
        
        results = manager.generate_with_repeats(
            items,
            num_repeats=3,
            repeat_strategy="temperature_schedule",
            temperature_schedule=temperature_schedule
        )
        
        assert len(captured_params) == 3
        assert captured_params[0]["temperature"] == 0.5
        assert captured_params[1]["temperature"] == 1.0
        assert captured_params[2]["temperature"] == 1.5
    
    def test_error_handling_skip(self, generation_config):
        """Test error handling with skip strategy"""
        # Create a model that fails
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Generation failed"))
        
        generation_config.error_handling = "skip"
        manager = GenerationManager(mock_model, generation_config)
        
        items = [{"idx": 0, "prompt": "Test"}]
        results = manager.generate_batch(items)
        
        assert len(results) == 1
        assert results[0]["response"] == ""
        assert "error" in results[0]
        assert results[0]["error"] == "Generation failed"
    
    def test_error_handling_retry(self, generation_config):
        """Test error handling with retry strategy"""
        mock_model = Mock()
        
        # Fail twice, then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return [{
                "prompt": args[0][0],
                "response": "Success",
                "tokens": 5,
                "latency": 0.1
            }]
        
        mock_model.generate = Mock(side_effect=side_effect)
        
        generation_config.error_handling = "retry"
        generation_config.max_retries = 3
        manager = GenerationManager(mock_model, generation_config)
        
        items = [{"idx": 0, "prompt": "Test"}]
        results = manager.generate_batch(items)
        
        assert len(results) == 1
        assert results[0]["response"] == "Success"
        assert call_count == 3  # Failed twice, succeeded on third try
    
    def test_metrics_tracking(self, mock_model, generation_config):
        """Test metrics tracking"""
        manager = GenerationManager(mock_model, generation_config)
        
        items = [
            {"idx": 0, "prompt": "Test 1"},
            {"idx": 1, "prompt": "Test 2"}
        ]
        
        manager.generate_batch(items)
        metrics = manager.get_metrics()
        
        assert metrics["total_prompts"] == 2
        assert metrics["successful_generations"] == 2
        assert metrics["failed_generations"] == 0
        assert metrics["total_tokens"] > 0
        assert metrics["total_time"] > 0
        assert "avg_tokens_per_generation" in metrics
        assert "avg_time_per_generation" in metrics
        assert "tokens_per_second" in metrics
    
    def test_progress_callback(self, mock_model, generation_config):
        """Test progress callback functionality"""
        manager = GenerationManager(mock_model, generation_config)
        
        progress_updates = []
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
        
        items = [
            {"idx": i, "prompt": f"Test {i}"}
            for i in range(5)
        ]
        
        manager.generate_batch(items, progress_callback=progress_callback)
        
        assert len(progress_updates) > 0
        # Last update should show all items completed
        assert progress_updates[-1] == (5, 5)
    
    def test_estimate_time_remaining(self, mock_model, generation_config):
        """Test time estimation"""
        manager = GenerationManager(mock_model, generation_config)
        
        # Generate some items to establish metrics
        items = [{"idx": i, "prompt": f"Test {i}"} for i in range(10)]
        manager.generate_batch(items)
        
        # Estimate time for remaining items
        estimated_time = manager.estimate_time_remaining(completed=10, total=20)
        
        assert estimated_time > 0  # Should have some estimate
    
    def test_seed_increment(self, mock_model):
        """Test seed incrementing for repeats"""
        config = GenerationConfig(seed_increment=10)
        manager = GenerationManager(mock_model, config)
        
        items = [{"idx": 0, "prompt": "Test"}]
        base_params = {"seed": 100}
        
        captured_params = []
        original_generate = manager.model.generate
        
        def mock_generate(prompts, sampling_params=None):
            captured_params.append(sampling_params.copy() if sampling_params else {})
            return original_generate(prompts, sampling_params)
        
        manager.model.generate = mock_generate
        
        results = manager.generate_with_repeats(
            items,
            num_repeats=3,
            repeat_strategy="independent",
            sampling_params=base_params
        )
        
        # Check that seeds were incremented
        assert captured_params[0]["seed"] == 100
        assert captured_params[1]["seed"] == 110
        assert captured_params[2]["seed"] == 120