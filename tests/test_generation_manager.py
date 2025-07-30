"""Tests for generation manager."""

import pytest
from unittest.mock import AsyncMock

from vllm_generator.models import GenerationManager
from vllm_generator.config.schemas import Config, DataConfig, ModelConfig, GenerationConfig
from pathlib import Path


class TestGenerationManager:
    """Test GenerationManager functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            data=DataConfig(
                input_path=Path("input.parquet"),
                output_path=Path("output.parquet")
            ),
            models=[
                ModelConfig(url="http://server1:8000", name="model1"),
                ModelConfig(url="http://server2:8000", name="model2")
            ],
            generation=GenerationConfig(
                num_samples=1,
                temperature=1.0,
                max_tokens=512
            )
        )
    
    def test_initialization(self, test_config):
        """Test generation manager initialization."""
        manager = GenerationManager(test_config)
        
        assert len(manager.clients) == 2
        assert "http://server1:8000/" in manager.clients
        assert "http://server2:8000/" in manager.clients
        assert manager.model_manager is not None
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, test_config):
        """Test health checking all endpoints."""
        manager = GenerationManager(test_config)
        
        # Mock health checks
        for client in manager.clients.values():
            client.health_check = AsyncMock(return_value=True)
        
        results = await manager.health_check_all()
        
        assert results["model1"] is True
        assert results["model2"] is True
        
        # Verify all clients were checked
        for client in manager.clients.values():
            client.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_single(self, test_config, mock_vllm_response):
        """Test single generation."""
        manager = GenerationManager(test_config)
        
        # Mock client generate method
        for client in manager.clients.values():
            client.generate = AsyncMock(return_value=mock_vllm_response)
        
        result = await manager.generate_single("Test prompt")
        
        assert result == mock_vllm_response
        
        # At least one client should have been called
        called = sum(1 for c in manager.clients.values() if c.generate.called)
        assert called == 1
    
    @pytest.mark.asyncio
    async def test_generate_single_with_samples(self, test_config, mock_multi_sample_response):
        """Test single generation with multiple samples."""
        test_config.generation.num_samples = 3
        manager = GenerationManager(test_config)
        
        # Mock client generate_samples method
        for client in manager.clients.values():
            client.generate_samples = AsyncMock(
                return_value=[mock_multi_sample_response] * 3
            )
        
        result = await manager.generate_single("Test prompt")
        
        assert "choices" in result
        assert len(result["choices"]) == 9  # 3 samples * 3 choices each
    
    @pytest.mark.asyncio
    async def test_generate_batch(self, test_config, mock_vllm_responses):
        """Test batch generation."""
        manager = GenerationManager(test_config)
        
        # Mock client methods
        for client in manager.clients.values():
            client.generate = AsyncMock(side_effect=mock_vllm_responses)
            client.generate_samples = AsyncMock()
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = await manager.generate_batch(prompts)
        
        assert len(results) == 3
        assert all("choices" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_generate_batch_with_progress(self, test_config, mock_vllm_response):
        """Test batch generation with progress callback."""
        manager = GenerationManager(test_config)
        
        # Mock client
        for client in manager.clients.values():
            client.generate = AsyncMock(return_value=mock_vllm_response)
        
        progress_counts = []
        def progress_callback(count):
            progress_counts.append(count)
        
        prompts = ["Prompt 1", "Prompt 2"]
        await manager.generate_batch(prompts, progress_callback=progress_callback)
        
        assert len(progress_counts) == 2
        assert all(c == 1 for c in progress_counts)
    
    def test_extract_texts_single_response(self, test_config, mock_vllm_responses):
        """Test extracting texts from single responses."""
        manager = GenerationManager(test_config)
        
        texts = manager.extract_texts_from_responses(mock_vllm_responses[:3])
        
        assert len(texts) == 3
        assert texts[0] == "Response 0: This is a test response"
        assert texts[1] == "Response 1: This is a test response"
        assert texts[2] == "Response 2: This is a test response"
    
    def test_extract_texts_multiple_samples(self, test_config):
        """Test extracting texts with multiple samples."""
        test_config.generation.num_samples = 2
        manager = GenerationManager(test_config)
        
        responses = [
            {
                "choices": [
                    {"text": "Sample 1a"},
                    {"text": "Sample 1b"}
                ]
            },
            {
                "choices": [
                    {"text": "Sample 2a"},
                    {"text": "Sample 2b"}
                ]
            }
        ]
        
        texts = manager.extract_texts_from_responses(responses)
        
        assert len(texts) == 2
        assert texts[0] == ["Sample 1a", "Sample 1b"]
        assert texts[1] == ["Sample 2a", "Sample 2b"]
    
    def test_extract_texts_with_errors(self, test_config):
        """Test extracting texts when some responses have errors."""
        manager = GenerationManager(test_config)
        
        responses = [
            {"choices": [{"text": "Success"}]},
            {"error": "Failed"},
            {"choices": []},  # Empty choices
        ]
        
        texts = manager.extract_texts_from_responses(responses)
        
        assert len(texts) == 3
        assert texts[0] == "Success"
        assert texts[1] == ""
        assert texts[2] == ""
    
    def test_get_statistics(self, test_config):
        """Test getting statistics."""
        manager = GenerationManager(test_config)
        
        # Set some stats on model manager
        manager.model_manager.endpoints[0].request_count = 10
        manager.model_manager.endpoints[0].error_count = 2
        
        stats = manager.get_statistics()
        
        assert stats["total_endpoints"] == 2
        assert stats["total_requests"] == 10
        assert stats["total_errors"] == 2