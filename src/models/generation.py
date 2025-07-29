import logging
import time
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .config import ModelConfig, GenerationConfig
from .vllm_wrapper import VLLMModel, BaseVLLMModel

logger = logging.getLogger(__name__)


class GenerationManager:
    """Manages the generation process with batching and error handling"""
    
    def __init__(
        self,
        model: BaseVLLMModel,
        generation_config: GenerationConfig,
        track_metrics: bool = True
    ):
        self.model = model
        self.config = generation_config
        self.track_metrics = track_metrics
        self.metrics = {
            "total_prompts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_tokens": 0,
            "total_time": 0,
            "retries": 0
        }
    
    def generate_batch(
        self,
        items: List[Dict[str, Any]],
        sampling_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of items"""
        results = []
        
        # Process in smaller batches if needed
        batch_size = self.config.batch_size
        num_batches = (len(items) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            
            # Extract prompts
            prompts = [item["prompt"] for item in batch_items]
            
            # Generate with error handling
            batch_results = self._generate_with_retry(prompts, sampling_params)
            
            # Combine with original items
            for i, (item, result) in enumerate(zip(batch_items, batch_results)):
                combined = {
                    **item,
                    **result,
                    "batch_idx": batch_idx
                }
                results.append(combined)
            
            # Update metrics
            if self.track_metrics:
                self._update_metrics(batch_results)
            
            # Progress callback
            if progress_callback:
                progress_callback(end_idx, len(items))
        
        return results
    
    def generate_with_repeats(
        self,
        items: List[Dict[str, Any]],
        num_repeats: int,
        repeat_strategy: str = "independent",
        temperature_schedule: Optional[List[float]] = None,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses per item"""
        all_results = []
        
        for repeat_id in range(num_repeats):
            logger.info(f"Generating repeat {repeat_id + 1}/{num_repeats}")
            
            # Adjust sampling params for this repeat
            repeat_params = self._get_repeat_params(
                repeat_id,
                repeat_strategy,
                temperature_schedule,
                sampling_params
            )
            
            # Generate
            repeat_results = self.generate_batch(items, repeat_params)
            
            # Add repeat information
            for result in repeat_results:
                result["repeat_id"] = repeat_id
            
            all_results.extend(repeat_results)
        
        return all_results
    
    def generate_parallel_batches(
        self,
        item_batches: List[List[Dict[str, Any]]],
        num_workers: int = 4,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple batches in parallel using threads"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for batch_idx, batch_items in enumerate(item_batches):
                future = executor.submit(
                    self.generate_batch,
                    batch_items,
                    sampling_params
                )
                future_to_batch[future] = batch_idx
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    if self.config.error_handling == "fail":
                        raise
        
        return all_results
    
    def _generate_with_retry(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Set timeout if specified
                if self.config.timeout_per_request:
                    # Would need to implement timeout logic here
                    # For now, just call generate
                    results = self.model.generate(prompts, sampling_params)
                else:
                    results = self.model.generate(prompts, sampling_params)
                
                # Success
                generation_time = time.time() - start_time
                for result in results:
                    result["generation_time"] = generation_time / len(results)
                
                return results
                
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if self.track_metrics:
                    self.metrics["retries"] += 1
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        if self.config.error_handling == "fail":
            raise last_error
        elif self.config.error_handling == "skip":
            # Return empty results
            logger.error(f"Skipping {len(prompts)} prompts due to generation failure")
            return [
                {
                    "prompt": prompt,
                    "response": "",
                    "error": str(last_error),
                    "tokens": 0,
                    "latency": 0
                }
                for prompt in prompts
            ]
        else:
            raise ValueError(f"Unknown error handling: {self.config.error_handling}")
    
    def _get_repeat_params(
        self,
        repeat_id: int,
        repeat_strategy: str,
        temperature_schedule: Optional[List[float]],
        base_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get sampling parameters for a specific repeat"""
        params = (base_params or {}).copy()
        
        if repeat_strategy == "temperature_schedule" and temperature_schedule:
            params["temperature"] = temperature_schedule[repeat_id]
        
        # Increment seed if using fixed seed
        if "seed" in params and params["seed"] is not None:
            params["seed"] = params["seed"] + repeat_id * self.config.seed_increment
        
        return params
    
    def _update_metrics(self, results: List[Dict[str, Any]]):
        """Update generation metrics"""
        self.metrics["total_prompts"] += len(results)
        
        for result in results:
            if result.get("error"):
                self.metrics["failed_generations"] += 1
            else:
                self.metrics["successful_generations"] += 1
                self.metrics["total_tokens"] += result.get("tokens", 0)
                self.metrics["total_time"] += result.get("latency", 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generation metrics"""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics["successful_generations"] > 0:
            metrics["avg_tokens_per_generation"] = (
                metrics["total_tokens"] / metrics["successful_generations"]
            )
            metrics["avg_time_per_generation"] = (
                metrics["total_time"] / metrics["successful_generations"]
            )
            metrics["tokens_per_second"] = (
                metrics["total_tokens"] / metrics["total_time"]
                if metrics["total_time"] > 0 else 0
            )
        
        return metrics
    
    def estimate_time_remaining(
        self,
        completed: int,
        total: int
    ) -> float:
        """Estimate time remaining based on current metrics"""
        if completed == 0 or self.metrics["successful_generations"] == 0:
            return 0
        
        avg_time = self.metrics["total_time"] / self.metrics["successful_generations"]
        remaining = total - completed
        
        return remaining * avg_time