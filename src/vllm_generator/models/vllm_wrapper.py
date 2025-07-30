"""vLLM client wrapper for API interactions."""

import asyncio
from typing import List, Dict, Any, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from ..config.schemas import ModelConfig, GenerationConfig, RetryConfig
from ..utils import get_logger


class VLLMClient:
    """Async client for vLLM server interactions."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        retry_config: RetryConfig
    ):
        """Initialize vLLM client."""
        self.model_config = model_config
        self.generation_config = generation_config
        self.retry_config = retry_config
        self.logger = get_logger(f"VLLMClient[{model_config.name or model_config.url}]")
        
        # Setup HTTP client
        self.client = httpx.AsyncClient(
            base_url=str(model_config.url),
            timeout=retry_config.timeout,
            headers=model_config.headers or {}
        )
        
        # API endpoints
        self.completions_endpoint = "/v1/completions"
        self.chat_endpoint = "/v1/chat/completions"
        self.health_endpoint = "/health"
        self.models_endpoint = "/v1/models"
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(get_logger("VLLMClient"), "WARNING")
    )
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self.client.get(self.health_endpoint)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the server."""
        try:
            response = await self.client.get(self.models_endpoint)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def _prepare_request(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Prepare request payload for vLLM."""
        # Handle temperature scheduling
        if isinstance(self.generation_config.temperature, list):
            temp = self.generation_config.temperature[sample_idx % len(self.generation_config.temperature)]
        else:
            temp = temperature or self.generation_config.temperature
        
        request = {
            "prompt": prompt,
            "max_tokens": self.generation_config.max_tokens,
            "temperature": temp,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
            "presence_penalty": self.generation_config.presence_penalty,
            "frequency_penalty": self.generation_config.frequency_penalty,
            "n": 1,  # Always generate 1 at a time for better control
        }
        
        if self.generation_config.stop_sequences:
            request["stop"] = self.generation_config.stop_sequences
        
        if self.generation_config.seed is not None:
            request["seed"] = self.generation_config.seed
        
        return request
    
    @retry(
        stop=lambda r: r.attempt_number > r.retry_object.retry_config.max_retries,
        wait=lambda r: wait_exponential(
            multiplier=r.retry_object.retry_config.retry_delay,
            max=r.retry_object.retry_config.retry_delay * r.retry_object.retry_config.backoff_factor ** 3
        )(r),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(get_logger("VLLMClient"), "WARNING")
    )
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Generate completion for a single prompt."""
        request = self._prepare_request(prompt, temperature, sample_idx)
        
        try:
            response = await self.client.post(
                self.completions_endpoint,
                json=request
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.generate(prompt, temperature, idx)
            for idx, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to generate for prompt {i}: {result}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def generate_samples(
        self,
        prompt: str,
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple samples for a single prompt."""
        tasks = [
            self.generate(prompt, sample_idx=i)
            for i in range(num_samples)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to generate sample {i}: {result}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results