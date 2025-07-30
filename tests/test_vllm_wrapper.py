"""Tests for vLLM client wrapper."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
import asyncio

from vllm_generator.models import VLLMClient
from vllm_generator.config.schemas import ModelConfig, GenerationConfig, RetryConfig


class TestVLLMClient:
    """Test VLLMClient functionality."""
    
    @pytest.fixture
    def client_config(self):
        """Create test client configuration."""
        return {
            "model_config": ModelConfig(
                url="http://localhost:8000",
                name="test_model"
            ),
            "generation_config": GenerationConfig(
                num_samples=1,
                temperature=1.0,
                max_tokens=512
            ),
            "retry_config": RetryConfig(
                max_retries=3,
                retry_delay=1.0,
                timeout=300.0
            )
        }
    
    def test_client_initialization(self, client_config):
        """Test client initialization."""
        client = VLLMClient(**client_config)
        
        assert client.model_config.name == "test_model"
        assert client.generation_config.temperature == 1.0
        assert client.retry_config.max_retries == 3
        assert client.completions_endpoint == "/v1/completions"
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client_config):
        """Test async context manager."""
        async with VLLMClient(**client_config) as client:
            assert client is not None
            assert hasattr(client, 'client')
    
    def test_prepare_request_basic(self, client_config):
        """Test basic request preparation."""
        client = VLLMClient(**client_config)
        
        request = client._prepare_request("Test prompt")
        
        assert request["prompt"] == "Test prompt"
        assert request["max_tokens"] == 512
        assert request["temperature"] == 1.0
        assert request["n"] == 1
    
    def test_prepare_request_with_temperature_schedule(self, client_config):
        """Test request preparation with temperature schedule."""
        client_config["generation_config"].num_samples = 3
        client_config["generation_config"].temperature = [0.5, 0.7, 0.9]
        
        client = VLLMClient(**client_config)
        
        # Different temperatures for different samples
        req1 = client._prepare_request("Test", sample_idx=0)
        assert req1["temperature"] == 0.5
        
        req2 = client._prepare_request("Test", sample_idx=1)
        assert req2["temperature"] == 0.7
        
        req3 = client._prepare_request("Test", sample_idx=2)
        assert req3["temperature"] == 0.9
    
    def test_prepare_request_with_stop_sequences(self, client_config):
        """Test request preparation with stop sequences."""
        client_config["generation_config"].stop_sequences = ["###", "END"]
        
        client = VLLMClient(**client_config)
        request = client._prepare_request("Test prompt")
        
        assert request["stop"] == ["###", "END"]
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client_config):
        """Test successful health check."""
        client = VLLMClient(**client_config)
        
        # Mock the HTTP client
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await client.health_check()
            
            assert result is True
            mock_get.assert_called_once_with("/health")
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client_config):
        """Test failed health check."""
        client = VLLMClient(**client_config)
        
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")
            
            result = await client.health_check()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, client_config):
        """Test listing models."""
        client = VLLMClient(**client_config)
        
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "model1"},
                    {"id": "model2"}
                ]
            }
            mock_response.raise_for_status = AsyncMock()
            mock_get.return_value = mock_response
            
            models = await client.list_models()
            
            assert models == ["model1", "model2"]
            mock_get.assert_called_once_with("/v1/models")
    
    @pytest.mark.asyncio
    async def test_generate_success(self, client_config, mock_vllm_response):
        """Test successful generation."""
        client = VLLMClient(**client_config)
        
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vllm_response
            mock_response.raise_for_status = AsyncMock()
            mock_post.return_value = mock_response
            
            result = await client.generate("Test prompt")
            
            assert result == mock_vllm_response
            mock_post.assert_called_once()
            
            # Check request payload
            call_args = mock_post.call_args
            assert call_args[0][0] == "/v1/completions"
            assert call_args[1]["json"]["prompt"] == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_generate_batch(self, client_config, mock_vllm_responses):
        """Test batch generation."""
        client = VLLMClient(**client_config)
        
        with patch.object(client, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = mock_vllm_responses[:3]
            
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await client.generate_batch(prompts)
            
            assert len(results) == 3
            assert all("choices" in r for r in results)
            assert mock_generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_generate_samples(self, client_config, mock_vllm_response):
        """Test generating multiple samples."""
        client_config["generation_config"].num_samples = 3
        client = VLLMClient(**client_config)
        
        with patch.object(client, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_vllm_response
            
            results = await client.generate_samples("Test prompt", num_samples=3)
            
            assert len(results) == 3
            assert mock_generate.call_count == 3
            
            # Verify sample indices
            for i, call in enumerate(mock_generate.call_args_list):
                assert call[1]["sample_idx"] == i