import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import pandas as pd

from .base import SimplePipeline
from .batch_processor import DistributedBatchProcessor
from ..data import DataLoader, DataProcessor, DataWriter
from ..models import ModelConfig, VLLMModel, VLLMServer
from ..tracking import GenerationTracker

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a worker process"""
    worker_id: int
    gpu_id: Optional[int]
    port: Optional[int]
    shard_path: Optional[str]
    output_path: str


class PipelineManager:
    """Manages parallel execution of generation pipelines"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: Dict[str, Any],
        parallel_mode: str = "single",
        num_workers: int = 1,
        worker_gpus: Optional[List[int]] = None,
        base_port: int = 8000
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self.parallel_mode = parallel_mode
        self.num_workers = num_workers
        self.worker_gpus = worker_gpus or list(range(num_workers))
        self.base_port = base_port
        
        # Coordinator components
        self.coordinator_port = base_port - 1
        self.work_queue = None
        self.result_queue = None
        
    def run(
        self,
        input_path: str,
        output_path: str,
        question_column: str = "question",
        output_format: str = "wide",
        **kwargs
    ) -> Dict[str, Any]:
        """Run the pipeline with specified parallelism"""
        logger.info(f"Running pipeline in {self.parallel_mode} mode with {self.num_workers} workers")
        
        if self.parallel_mode == "single":
            return self._run_single(input_path, output_path, question_column, output_format, **kwargs)
        elif self.parallel_mode == "multi_server":
            return self._run_multi_server(input_path, output_path, question_column, output_format, **kwargs)
        elif self.parallel_mode == "ray":
            return self._run_ray(input_path, output_path, question_column, output_format, **kwargs)
        else:
            raise ValueError(f"Unknown parallel mode: {self.parallel_mode}")
    
    def _run_single(
        self,
        input_path: str,
        output_path: str,
        question_column: str,
        output_format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run single-process pipeline"""
        # Create components
        data_loader = DataLoader(input_path, question_column)
        data_processor = DataProcessor(**kwargs)
        data_writer = DataWriter(output_format=output_format)
        tracker = GenerationTracker(output_dir=Path(output_path).parent)
        
        # Create pipeline
        from ..models.config import GenerationConfig
        gen_config = GenerationConfig.from_dict(self.generation_config)
        
        pipeline = SimplePipeline(
            self.model_config,
            gen_config,
            data_loader,
            data_processor,
            data_writer,
            tracker
        )
        
        # Run
        pipeline.initialize()
        try:
            result = pipeline.run()
            return result
        finally:
            pipeline.shutdown()
    
    def _run_multi_server(
        self,
        input_path: str,
        output_path: str,
        question_column: str,
        output_format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run multi-server parallel pipeline"""
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shards
        logger.info("Creating data shards")
        data_loader = DataLoader(input_path, question_column)
        shard_dir = output_dir / "shards"
        shards = data_loader.create_shards(
            self.num_workers,
            output_dir=str(shard_dir),
            strategy=kwargs.get("sharding_strategy", "contiguous")
        )
        
        # Start vLLM servers
        servers = []
        logger.info("Starting vLLM servers")
        
        try:
            for i in range(self.num_workers):
                # Configure GPU
                gpu_id = self.worker_gpus[i] if i < len(self.worker_gpus) else i
                config = self.model_config.__class__(**self.model_config.to_dict())
                config.device = f"cuda:{gpu_id}"
                
                # Start server
                port = self.base_port + i
                server = VLLMServer(config, port)
                server.start()
                servers.append(server)
                logger.info(f"Started vLLM server {i} on port {port}")
            
            # Process shards in parallel
            logger.info("Processing shards in parallel")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit work to each server
                future_to_worker = {}
                for i, (shard, server) in enumerate(zip(shards, servers)):
                    worker_config = WorkerConfig(
                        worker_id=i,
                        gpu_id=self.worker_gpus[i] if i < len(self.worker_gpus) else i,
                        port=server.port,
                        shard_path=shard.get("path"),
                        output_path=str(output_dir / f"worker_{i}_output.parquet")
                    )
                    
                    future = executor.submit(
                        self._process_shard_with_server,
                        worker_config,
                        server,
                        question_column,
                        output_format,
                        **kwargs
                    )
                    future_to_worker[future] = i
                
                # Collect results
                worker_results = []
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        result = future.result()
                        worker_results.append(result)
                        logger.info(f"Worker {worker_id} completed")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed: {e}")
                        if kwargs.get("worker_failure_mode", "retry") == "fail":
                            raise
                            
        except Exception as e:
            logger.error(f"Multi-server pipeline failed: {e}")
            # Always shutdown servers on error
            for server in servers:
                try:
                    logger.info(f"Shutting down server on port {server.port}")
                    server.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error shutting down server: {shutdown_error}")
            raise
        
        # Shutdown servers normally
        logger.info("Shutting down vLLM servers")
        for server in servers:
            try:
                server.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down server on port {server.port}: {e}")
        
        # Merge results
        logger.info("Merging worker outputs")
        result_files = [r["output_path"] for r in worker_results]
        data_writer = DataWriter(output_format=output_format)
        merged_df = data_writer.merge_results(
            result_files,
            output_path,
            strategy=kwargs.get("result_aggregation", "sequential")
        )
        
        # Calculate total metrics
        total_metrics = self._aggregate_metrics(worker_results)
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        return {
            "output_path": output_path,
            "metrics": total_metrics,
            "total_samples": len(merged_df),
            "num_workers": self.num_workers,
            "total_time": total_time
        }
    
    def _process_shard_with_server(
        self,
        worker_config: WorkerConfig,
        server: VLLMServer,
        question_column: str,
        output_format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a shard using a vLLM server"""
        logger.info(f"Worker {worker_config.worker_id} processing shard")
        
        # Load shard
        shard_df = pd.read_parquet(worker_config.shard_path)
        
        # Process data
        data_processor = DataProcessor(**kwargs)
        items = data_processor.process_batch(shard_df, question_column)
        
        # Generate using server
        results = []
        batch_size = self.generation_config.get("batch_size", 32)
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            prompts = [item["prompt"] for item in batch]
            
            # Call server
            batch_results = server.generate(prompts)
            
            # Add indices
            for j, result in enumerate(batch_results):
                result["idx"] = batch[j]["idx"]
                results.append(result)
        
        # Write results
        data_writer = DataWriter(output_format=output_format)
        data_writer.write_results(
            shard_df,
            results,
            worker_config.output_path,
            num_repeats=self.generation_config.get("num_repeats", 1)
        )
        
        return {
            "worker_id": worker_config.worker_id,
            "output_path": worker_config.output_path,
            "samples_processed": len(shard_df),
            "results_generated": len(results)
        }
    
    def _run_ray(
        self,
        input_path: str,
        output_path: str,
        question_column: str,
        output_format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run Ray-based distributed pipeline"""
        try:
            import ray
        except ImportError:
            raise ImportError("Ray is required for ray parallel mode. Install with: pip install ray")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address=kwargs.get("ray_address", "auto"))
        
        # Create Ray remote functions
        @ray.remote(num_gpus=kwargs.get("ray_num_gpus", 1))
        class RayWorker:
            def __init__(self, model_config, generation_config):
                from ..models import GenerationManager
                from ..models.config import GenerationConfig
                
                self.model = VLLMModel(model_config)
                gen_config = GenerationConfig.from_dict(generation_config)
                self.generation_manager = GenerationManager(self.model, gen_config)
            
            def process_batch(self, items):
                return self.generation_manager.generate_batch(items)
        
        # Create workers
        workers = [
            RayWorker.remote(self.model_config, self.generation_config)
            for _ in range(self.num_workers)
        ]
        
        # Load and distribute data
        data_loader = DataLoader(input_path, question_column)
        df = data_loader.load()
        
        data_processor = DataProcessor(**kwargs)
        all_items = data_processor.process_batch(df, question_column)
        
        # Distribute work
        batch_processor = DistributedBatchProcessor(
            self.num_workers,
            batch_size=self.generation_config.get("batch_size", 32),
            strategy="balanced"
        )
        
        worker_items = batch_processor.distribute_work(all_items)
        
        # Submit work to Ray
        futures = []
        for worker, items in zip(workers, worker_items):
            if items:  # Only submit if there's work
                batches = batch_processor.create_batches(items)
                for batch in batches:
                    future = worker.process_batch.remote(batch)
                    futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            batch_results = ray.get(future)
            results.extend(batch_results)
        
        # Write results
        data_writer = DataWriter(output_format=output_format)
        _ = data_writer.write_results(
            df,
            results,
            output_path,
            num_repeats=self.generation_config.get("num_repeats", 1)
        )
        
        # Shutdown Ray if we initialized it
        if kwargs.get("ray_address", "auto") == "auto":
            ray.shutdown()
        
        return {
            "output_path": output_path,
            "total_samples": len(df),
            "results_generated": len(results),
            "num_workers": self.num_workers
        }
    
    def _aggregate_metrics(self, worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple workers"""
        total_metrics = {
            "total_samples": sum(r.get("samples_processed", 0) for r in worker_results),
            "total_results": sum(r.get("results_generated", 0) for r in worker_results),
            "num_workers": len(worker_results)
        }
        return total_metrics