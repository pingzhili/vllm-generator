"""Generation manager for handling text generation with vLLM."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

from .vllm_wrapper import VLLMClient
from .config import ModelManager, ModelEndpoint
from ..config.schemas import Config, ModelConfig
from ..data.processor import DataProcessor
from ..utils import get_logger, distribute_items


class GenerationManager:
    """Manage text generation across multiple vLLM endpoints."""
    
    def __init__(self, config: Config):
        """Initialize generation manager."""
        self.config = config
        self.logger = get_logger("GenerationManager")
        self.model_manager = ModelManager(config.models)
        self.data_processor = DataProcessor()
        
        # Create clients for each endpoint
        self.clients: Dict[str, VLLMClient] = {}
        for model_config in config.models:
            client = VLLMClient(
                model_config=model_config,
                generation_config=config.generation,
                retry_config=config.retry
            )
            self.clients[str(model_config.url)] = client
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close all clients."""
        for client in self.clients.values():
            await client.close()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all endpoints."""
        results = {}
        
        for endpoint in self.model_manager.endpoints:
            client = self.clients[endpoint.url]
            is_healthy = await client.health_check()
            
            if is_healthy:
                self.model_manager.mark_healthy(endpoint)
            else:
                self.model_manager.mark_unhealthy(endpoint)
            
            results[endpoint.name] = is_healthy
        
        return results
    
    async def generate_single(
        self,
        prompt: str,
        endpoint: Optional[ModelEndpoint] = None
    ) -> Dict[str, Any]:
        """Generate response for a single prompt."""
        if endpoint is None:
            endpoint = self.model_manager.get_endpoint()
        
        client = self.clients[endpoint.url]
        
        try:
            if self.config.generation.num_samples > 1:
                # Generate multiple samples
                results = await client.generate_samples(
                    prompt,
                    self.config.generation.num_samples
                )
                # Combine results
                combined = {
                    "choices": [],
                    "model": endpoint.name
                }
                for result in results:
                    if "choices" in result:
                        combined["choices"].extend(result["choices"])
                return combined
            else:
                # Generate single sample
                return await client.generate(prompt)
        
        except Exception as e:
            self.model_manager.mark_unhealthy(endpoint)
            raise
    
    async def generate_batch(
        self,
        prompts: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts."""
        start_time = time.time()
        
        # Distribute prompts across workers
        num_workers = min(self.config.processing.num_workers, len(prompts))
        prompt_chunks = distribute_items(prompts, num_workers)
        
        self.logger.info(
            f"Processing {len(prompts)} prompts with {num_workers} workers"
        )
        
        # Process chunks in parallel
        tasks = []
        for i, chunk in enumerate(prompt_chunks):
            endpoint = self.model_manager.get_endpoint()
            task = self._process_chunk(chunk, endpoint, progress_callback)
            tasks.append(task)
        
        # Gather results
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                self.logger.error(f"Chunk processing failed: {chunk_result}")
                # Add error results for the chunk
                results.extend([{"error": str(chunk_result)}] * len(prompts) // num_workers)
            else:
                results.extend(chunk_result)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"Processed {len(prompts)} prompts in {elapsed:.2f}s "
            f"({len(prompts) / elapsed:.2f} prompts/s)"
        )
        
        return results
    
    async def _process_chunk(
        self,
        prompts: List[str],
        endpoint: ModelEndpoint,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process a chunk of prompts."""
        client = self.clients[endpoint.url]
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                if self.config.generation.num_samples > 1:
                    # Generate multiple samples
                    samples = await client.generate_samples(
                        prompt,
                        self.config.generation.num_samples
                    )
                    # Combine samples
                    combined = {
                        "choices": [],
                        "model": endpoint.name
                    }
                    for sample in samples:
                        if "choices" in sample:
                            combined["choices"].extend(sample["choices"])
                    results.append(combined)
                else:
                    # Generate single sample
                    result = await client.generate(prompt)
                    results.append(result)
                
                if progress_callback:
                    progress_callback(1)
            
            except Exception as e:
                self.logger.error(f"Failed to generate for prompt: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def extract_texts_from_responses(
        self,
        responses: List[Dict[str, Any]]
    ) -> List[Union[str, List[str]]]:
        """Extract text from vLLM responses."""
        texts = []
        
        for response in responses:
            if "error" in response:
                if self.config.generation.num_samples > 1:
                    texts.append([""] * self.config.generation.num_samples)
                else:
                    texts.append("")
            elif "choices" in response:
                if self.config.generation.num_samples > 1:
                    # Multiple samples
                    sample_texts = []
                    for choice in response["choices"]:
                        text = choice.get("text", "")
                        sample_texts.append(text)
                    texts.append(sample_texts)
                else:
                    # Single sample
                    text = response["choices"][0].get("text", "") if response["choices"] else ""
                    texts.append(text)
            else:
                if self.config.generation.num_samples > 1:
                    texts.append([""] * self.config.generation.num_samples)
                else:
                    texts.append("")
        
        return texts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.model_manager.get_statistics()