"""Pipeline manager for orchestrating the generation process."""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from .base import Pipeline, PipelineStep
from .batch_processor import BatchProcessor
from ..config.schemas import Config
from ..models import GenerationManager
from ..utils import get_logger, setup_logger


class GenerationPipeline:
    """Main pipeline for data generation."""
    
    def __init__(self, config: Config):
        """Initialize generation pipeline."""
        self.config = config
        self.logger = get_logger("GenerationPipeline")
        
        # Setup logging
        setup_logger(config.logging)
        
        # Initialize components
        self.batch_processor = BatchProcessor(config)
        self.generation_manager = None
    
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        self.logger.info("Initializing pipeline...")
        
        # Create generation manager
        self.generation_manager = GenerationManager(self.config)
        
        # Health check all endpoints
        health_status = await self.generation_manager.health_check_all()
        
        healthy_count = sum(1 for status in health_status.values() if status)
        self.logger.info(
            f"Health check complete: {healthy_count}/{len(health_status)} endpoints healthy"
        )
        
        if healthy_count == 0:
            raise RuntimeError("No healthy endpoints available")
    
    async def run(
        self,
        dry_run: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Run the generation pipeline."""
        try:
            # Initialize if not already done
            if self.generation_manager is None:
                await self.initialize()
            
            # Process data
            self.logger.info("Starting data processing...")
            
            async with self.generation_manager:
                results = await self.batch_processor.process(
                    self.generation_manager,
                    dry_run=dry_run,
                    progress_bar=progress_bar
                )
            
            self.logger.info("Pipeline completed successfully")
            return results
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    async def validate(self) -> bool:
        """Validate pipeline configuration."""
        self.logger.info("Validating pipeline configuration...")
        
        # Check input file exists
        if not self.config.data.input_path.exists():
            self.logger.error(f"Input file not found: {self.config.data.input_path}")
            return False
        
        # Check output directory is writable
        output_dir = self.config.data.output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True)
            except Exception as e:
                self.logger.error(f"Cannot create output directory: {e}")
                return False
        
        # Initialize and check models
        try:
            await self.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    async def list_models(self) -> Dict[str, List[str]]:
        """List available models from all endpoints."""
        if self.generation_manager is None:
            await self.initialize()
        
        model_lists = {}
        
        async with self.generation_manager:
            for endpoint in self.generation_manager.model_manager.endpoints:
                client = self.generation_manager.clients[endpoint.url]
                models = await client.list_models()
                model_lists[endpoint.name] = models
        
        return model_lists


async def run_pipeline(
    config: Config,
    dry_run: bool = False,
    progress_bar: bool = True
) -> Dict[str, Any]:
    """Convenience function to run the pipeline."""
    pipeline = GenerationPipeline(config)
    return await pipeline.run(dry_run=dry_run, progress_bar=progress_bar)