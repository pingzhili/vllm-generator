"""vLLM client using OpenAI API for interactions."""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import openai

from ..config.schemas import ModelConfig, GenerationConfig, RetryConfig
from ..utils import get_logger


class VLLMClient:
    """Client for vLLM server using OpenAI API."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        retry_config: RetryConfig
    ):
        """Initialize vLLM client with OpenAI API."""
        self.model_config = model_config
        self.generation_config = generation_config
        self.retry_config = retry_config
        self.logger = get_logger(f"VLLMClient[{str(model_config.url)}]")
        
        # Setup OpenAI client for vLLM
        base_url = str(model_config.url).rstrip('/') + '/v1'
        api_key = model_config.api_key or "EMPTY"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=retry_config.timeout
        )
        
        # Store default headers if any
        self.headers = model_config.headers or {}
        
        self.logger.info(f"Initialized OpenAI client for vLLM at {base_url}")
        self.logger.info(f"Generation config: {generation_config}")
    
    def close(self):
        """Close the client (OpenAI client handles connection pooling)."""
        if hasattr(self.client, 'close'):
            self.client.close()
    
    def health_check(self) -> bool:
        """Check if vLLM server is healthy by making a simple API call."""
        try:
            # Try to list models as a health check
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models on the server."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def _get_model_name(self) -> str:
        """Get the model name to use for requests."""
        # Try to get the first available model
        models = self.list_models()
        if models:
            return models[0]
        else:
            # Fallback to a common model name - vLLM will use whatever model is loaded
            return "gpt-3.5-turbo"  # OpenAI client requires a model name
    
    def _prepare_chat_completion_params(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Prepare parameters for chat completion request."""
        # Handle temperature scheduling
        if isinstance(self.generation_config.temperature, list):
            temp = self.generation_config.temperature[sample_idx % len(self.generation_config.temperature)]
        else:
            temp = temperature or self.generation_config.temperature
        
        # Prepare base parameters
        params = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.generation_config.max_tokens,
            "temperature": temp,
            "top_p": self.generation_config.top_p,
            "presence_penalty": self.generation_config.presence_penalty,
            "frequency_penalty": self.generation_config.frequency_penalty,
            "n": 1,  # Always generate 1 at a time for better control
        }
        
        # Add stop sequences if specified
        if self.generation_config.stop_sequences:
            params["stop"] = self.generation_config.stop_sequences
        
        # Add seed if specified
        if self.generation_config.seed is not None:
            params["seed"] = self.generation_config.seed
        
        # Prepare extra_body for vLLM-specific parameters
        extra_body = {}
        
        # Add top_k to extra_body (vLLM-specific)
        if self.generation_config.top_k > 0:
            extra_body["top_k"] = self.generation_config.top_k
        
        # Add enable_thinking to chat_template_kwargs
        if self.generation_config.enable_thinking is not None:
            if "chat_template_kwargs" not in extra_body:
                extra_body["chat_template_kwargs"] = {}
            extra_body["chat_template_kwargs"]["enable_thinking"] = self.generation_config.enable_thinking
            self.logger.debug(f"Setting enable_thinking to {self.generation_config.enable_thinking}")
        

        # Add extra_body to params if not empty
        if extra_body:
            params["extra_body"] = extra_body

        return params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError
        )),
        before_sleep=before_sleep_log(get_logger("VLLMClient.retry"), "WARNING")
    )
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Generate completion for a single prompt using chat completions."""
        params = self._prepare_chat_completion_params(prompt, temperature, sample_idx)
        
        try:
            response = self.client.chat.completions.create(**params)
            
            # Convert OpenAI response to our expected format
            return {
                "choices": [{
                    "text": response.choices[0].message.content or "",
                    "thinking_text": response.choices[0].message.reasoning_content or "",
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response.usage else {},
                "model": response.model,
                "id": response.id,
                "created": response.created
            }
            
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
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
                # Return error in expected format
                results.append({
                    "choices": [{"text": "", "finish_reason": "error", "index": 0}],
                    "error": str(e)
                })
        
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
                # Return error in expected format
                results.append({
                    "choices": [{"text": "", "finish_reason": "error", "index": 0}],
                    "error": str(e)
                })
        
        return results