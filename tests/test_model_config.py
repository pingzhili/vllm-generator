"""Tests for model configuration and management."""

from vllm_generator.models import ModelManager, ModelEndpoint
from vllm_generator.config.schemas import ModelConfig


class TestModelEndpoint:
    """Test ModelEndpoint functionality."""
    
    def test_model_endpoint_creation(self):
        """Test creating model endpoint."""
        endpoint = ModelEndpoint(
            url="http://localhost:8000",
            name="test_model",
            api_key="secret",
            headers={"Authorization": "Bearer token"},
            weight=2.0
        )
        
        assert endpoint.url == "http://localhost:8000"
        assert endpoint.name == "test_model"
        assert endpoint.api_key == "secret"
        assert endpoint.headers == {"Authorization": "Bearer token"}
        assert endpoint.weight == 2.0
        assert endpoint.is_healthy is True
        assert endpoint.request_count == 0
        assert endpoint.error_count == 0


class TestModelManager:
    """Test ModelManager functionality."""
    
    def test_initialization(self):
        """Test model manager initialization."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        assert len(manager.endpoints) == 2
        assert manager.endpoints[0].name == "model1"
        assert manager.endpoints[1].name == "model2"
    
    def test_get_endpoint_round_robin(self):
        """Test round-robin endpoint selection."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        # First round
        ep1 = manager.get_endpoint(strategy="round_robin")
        assert ep1.request_count == 1
        
        ep2 = manager.get_endpoint(strategy="round_robin")
        assert ep2.request_count == 1
        
        # Second round - should go back to first
        ep3 = manager.get_endpoint(strategy="round_robin")
        assert ep3 == ep1
        assert ep3.request_count == 2
    
    def test_get_endpoint_random(self):
        """Test random endpoint selection."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        # Random selection - just verify it returns valid endpoint
        endpoint = manager.get_endpoint(strategy="random")
        assert endpoint in manager.endpoints
        assert endpoint.request_count == 1
    
    def test_get_endpoint_unhealthy(self):
        """Test endpoint selection with unhealthy endpoints."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        # Mark first endpoint as unhealthy
        manager.mark_unhealthy(manager.endpoints[0])
        
        # Should only return healthy endpoint
        for _ in range(5):
            endpoint = manager.get_endpoint()
            assert endpoint.name == "model2"
            assert endpoint.is_healthy is True
    
    def test_mark_unhealthy_healthy(self):
        """Test marking endpoints as unhealthy/healthy."""
        configs = [ModelConfig(url="http://server1:8000", name="model1")]
        
        manager = ModelManager(configs)
        endpoint = manager.endpoints[0]
        
        assert endpoint.is_healthy is True
        assert endpoint.error_count == 0
        
        # Mark unhealthy
        manager.mark_unhealthy(endpoint)
        assert endpoint.is_healthy is False
        assert endpoint.error_count == 1
        
        # Mark healthy again
        manager.mark_healthy(endpoint)
        assert endpoint.is_healthy is True
        assert endpoint.error_count == 1  # Error count not reset
    
    def test_get_statistics(self):
        """Test getting endpoint statistics."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        # Simulate some requests
        ep1 = manager.endpoints[0]
        ep2 = manager.endpoints[1]
        
        ep1.request_count = 10
        ep1.error_count = 2
        ep2.request_count = 15
        ep2.error_count = 1
        
        stats = manager.get_statistics()
        
        assert stats["total_endpoints"] == 2
        assert stats["healthy_endpoints"] == 2
        assert stats["total_requests"] == 25
        assert stats["total_errors"] == 3
        
        # Check individual endpoint stats
        ep1_stats = stats["endpoints"][0]
        assert ep1_stats["name"] == "model1"
        assert ep1_stats["request_count"] == 10
        assert ep1_stats["error_count"] == 2
        assert ep1_stats["error_rate"] == 0.2
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        configs = [
            ModelConfig(url="http://server1:8000", name="model1"),
            ModelConfig(url="http://server2:8000", name="model2"),
        ]
        
        manager = ModelManager(configs)
        
        # Set some counts
        manager.endpoints[0].request_count = 10
        manager.endpoints[0].error_count = 2
        manager.endpoints[1].request_count = 15
        manager.endpoints[1].error_count = 3
        
        # Reset
        manager.reset_statistics()
        
        # Verify all reset
        for endpoint in manager.endpoints:
            assert endpoint.request_count == 0
            assert endpoint.error_count == 0