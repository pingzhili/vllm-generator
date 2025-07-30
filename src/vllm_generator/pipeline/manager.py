"""Pipeline manager for orchestrating the generation process."""

from typing import Dict, Any, List

from .simple_processor import SimpleProcessor
from ..config.schemas import Config
from ..models import VLLMClient
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
        self.processor = SimpleProcessor(config)
        self.vllm_client = None
    
    def initialize(self) -> None:
        """Initialize pipeline components."""
        self.logger.info("Initializing pipeline...")
        
        # Create vLLM client
        self.vllm_client = VLLMClient(
            self.config.model,
            self.config.generation,
            self.config.retry
        )
        
        # Health check
        if self.vllm_client.health_check():
            self.logger.info(f"vLLM server at {self.config.model.url} is healthy")
        else:
            raise RuntimeError(f"vLLM server at {self.config.model.url} is not healthy")
    
    def run(
        self,
        dry_run: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Run the generation pipeline."""
        try:
            # Initialize if not already done
            if self.vllm_client is None:
                self.initialize()
            
            # Process data
            self.logger.info("Starting data processing...")
            
            results = self.processor.process(
                self.vllm_client,
                dry_run=dry_run,
                progress_bar=progress_bar
            )
            
            self.logger.info("Pipeline completed successfully")
            return results
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Clean up
            if self.vllm_client:
                self.vllm_client.close()
    
    def validate(self) -> bool:
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
            self.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            return False
        finally:
            if self.vllm_client:
                self.vllm_client.close()
        
        self.logger.info("Configuration validation passed")
        return True
    
    def list_models(self) -> List[str]:
        """List available models from the endpoint."""
        if self.vllm_client is None:
            self.initialize()
        
        try:
            models = self.vllm_client.list_models()
            return models
        finally:
            if self.vllm_client:
                self.vllm_client.close()


def run_pipeline(
    config: Config,
    dry_run: bool = False,
    progress_bar: bool = True
) -> Dict[str, Any]:
    """Convenience function to run the pipeline."""
    pipeline = GenerationPipeline(config)
    return pipeline.run(dry_run=dry_run, progress_bar=progress_bar)