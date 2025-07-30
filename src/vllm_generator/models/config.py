"""Model configuration and management."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

from ..config.schemas import ModelConfig
from ..utils import get_logger


@dataclass
class ModelEndpoint:
    """Represents a single model endpoint."""
    
    url: str
    name: str
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    weight: float = 1.0
    is_healthy: bool = True
    request_count: int = 0
    error_count: int = 0


class ModelManager:
    """Manage multiple model endpoints with load balancing."""
    
    def __init__(self, model_configs: List[ModelConfig]):
        """Initialize model manager."""
        self.logger = get_logger("ModelManager")
        self.endpoints = []
        
        for config in model_configs:
            endpoint = ModelEndpoint(
                url=str(config.url),
                name=config.name or str(config.url),
                api_key=config.api_key,
                headers=config.headers
            )
            self.endpoints.append(endpoint)
        
        self.logger.info(f"Initialized with {len(self.endpoints)} endpoints")
    
    def get_endpoint(self, strategy: str = "round_robin") -> ModelEndpoint:
        """Get next endpoint based on strategy."""
        healthy_endpoints = [ep for ep in self.endpoints if ep.is_healthy]
        
        if not healthy_endpoints:
            self.logger.warning("No healthy endpoints available, using all endpoints")
            healthy_endpoints = self.endpoints
        
        if strategy == "round_robin":
            # Simple round-robin based on request count
            endpoint = min(healthy_endpoints, key=lambda x: x.request_count)
        elif strategy == "random":
            endpoint = random.choice(healthy_endpoints)
        elif strategy == "weighted":
            # Weighted random selection
            weights = [ep.weight for ep in healthy_endpoints]
            endpoint = random.choices(healthy_endpoints, weights=weights)[0]
        else:
            endpoint = healthy_endpoints[0]
        
        endpoint.request_count += 1
        return endpoint
    
    def mark_unhealthy(self, endpoint: ModelEndpoint) -> None:
        """Mark endpoint as unhealthy."""
        endpoint.is_healthy = False
        endpoint.error_count += 1
        self.logger.warning(f"Marked endpoint {endpoint.name} as unhealthy")
    
    def mark_healthy(self, endpoint: ModelEndpoint) -> None:
        """Mark endpoint as healthy."""
        endpoint.is_healthy = True
        self.logger.info(f"Marked endpoint {endpoint.name} as healthy")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all endpoints."""
        stats = {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": sum(1 for ep in self.endpoints if ep.is_healthy),
            "total_requests": sum(ep.request_count for ep in self.endpoints),
            "total_errors": sum(ep.error_count for ep in self.endpoints),
            "endpoints": []
        }
        
        for endpoint in self.endpoints:
            stats["endpoints"].append({
                "name": endpoint.name,
                "url": endpoint.url,
                "is_healthy": endpoint.is_healthy,
                "request_count": endpoint.request_count,
                "error_count": endpoint.error_count,
                "error_rate": endpoint.error_count / max(endpoint.request_count, 1)
            })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset request and error counts."""
        for endpoint in self.endpoints:
            endpoint.request_count = 0
            endpoint.error_count = 0
        self.logger.info("Reset endpoint statistics")