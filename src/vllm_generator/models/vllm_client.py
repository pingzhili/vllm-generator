"""Synchronous vLLM client for API interactions."""

from typing import List, Dict, Any, Optional, Union
import requests
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
    """Synchronous client for vLLM server interactions."""
    
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
        self.logger = get_logger(f"VLLMClient[{str(model_config.url)}]")
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update(model_config.headers or {})
        if model_config.api_key:
            self.session.headers["Authorization"] = f"Bearer {model_config.api_key}"
        
        # Base URL
        self.base_url = str(model_config.url).rstrip('/')
        
        # API endpoints
        self.completions_endpoint = f"{self.base_url}/v1/completions"
        self.health_endpoint = f"{self.base_url}/health"
        self.models_endpoint = f"{self.base_url}/v1/models"
    
    def close(self):
        """Close HTTP session."""
        self.session.close()
    
    def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = self.session.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models on the server."""
        try:
            response = self.session.get(self.models_endpoint, timeout=10)
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
            "presence_penalty": self.generation_config.presence_penalty,
            "frequency_penalty": self.generation_config.frequency_penalty,
            "n": 1,  # Always generate 1 at a time for better control
        }
        
        # Only add top_k to main request if it's not -1
        if self.generation_config.top_k > 0:
            request["top_k"] = self.generation_config.top_k
        
        if self.generation_config.stop_sequences:
            request["stop"] = self.generation_config.stop_sequences
        
        if self.generation_config.seed is not None:
            request["seed"] = self.generation_config.seed
        
        # Prepare extra_body
        extra_body = {}
        
        # Add enable_thinking to chat_template_kwargs
        if self.generation_config.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.generation_config.enable_thinking
            }
            self.logger.debug(f"Setting enable_thinking to {self.generation_config.enable_thinking}")
        
        # Merge with any custom extra_body from config
        if self.generation_config.extra_body:
            for key, value in self.generation_config.extra_body.items():
                if key == "chat_template_kwargs" and "chat_template_kwargs" in extra_body:
                    # Merge chat_template_kwargs
                    extra_body["chat_template_kwargs"].update(value)
                else:
                    extra_body[key] = value
        
        # Add extra_body to request if not empty
        if extra_body:
            request["extra_body"] = extra_body
        
        return request
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        before_sleep=before_sleep_log(get_logger("VLLMClient.retry"), "WARNING")
    )
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Generate completion for a single prompt."""
        request = self._prepare_request(prompt, temperature, sample_idx)
        
        try:
            response = self.session.post(
                self.completions_endpoint,
                json=request,
                timeout=self.retry_config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate completions for multiple prompts sequentially."""
        results = []
        
        for idx, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, temperature, idx)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate for prompt {idx}: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def generate_samples(
        self,
        prompt: str,
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple samples for a single prompt."""
        results = []
        
        for i in range(num_samples):
            try:
                result = self.generate(prompt, sample_idx=i)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate sample {i}: {e}")
                results.append({"error": str(e)})
        
        return results