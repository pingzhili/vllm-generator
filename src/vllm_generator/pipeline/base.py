import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..data import DataLoader, DataProcessor, DataWriter
from ..models import ModelConfig, VLLMModel, GenerationManager
from ..models.config import GenerationConfig
from ..tracking import GenerationTracker

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Base class for generation pipelines"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        data_loader: DataLoader,
        data_processor: DataProcessor,
        data_writer: DataWriter,
        tracker: Optional[GenerationTracker] = None
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.data_writer = data_writer
        self.tracker = tracker or GenerationTracker()
        
        # Initialize model and generation manager
        self.model = None
        self.generation_manager = None
        
    def initialize(self):
        """Initialize the pipeline components"""
        logger.info("Initializing pipeline")
        
        # Initialize model
        self.model = VLLMModel(self.model_config)
        
        # Set tokenizer on data processor if using chat template
        if hasattr(self.data_processor, 'use_chat_template') and self.data_processor.use_chat_template:
            if hasattr(self.model, 'get_tokenizer'):
                tokenizer = self.model.get_tokenizer()
                self.data_processor.set_tokenizer(tokenizer)
            else:
                logger.warning("Model does not provide tokenizer access for chat template")
        
        # Initialize generation manager
        self.generation_manager = GenerationManager(
            self.model,
            self.generation_config,
            track_metrics=True
        )
        
        # Validate configuration
        self.model_config.validate()
        
        logger.info("Pipeline initialized successfully")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the generation pipeline"""
        pass
    
    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down pipeline")
        
        if self.model:
            self.model.shutdown()
        
        # Save final metrics
        if self.tracker:
            self.tracker.save_final_metrics()
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if resuming"""
        if self.generation_config.resume_from_checkpoint:
            try:
                checkpoint = DataWriter.load_checkpoint(
                    self.generation_config.resume_from_checkpoint
                )
                logger.info(f"Loaded checkpoint: {checkpoint.get('completed_items', 0)} items completed")
                return checkpoint
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None
        return None
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save checkpoint"""
        if self.tracker:
            checkpoint_path = self.tracker.get_checkpoint_path()
            self.data_writer.write_checkpoint(checkpoint_data, checkpoint_path)
    
    def _should_save_checkpoint(self, current_idx: int) -> bool:
        """Check if checkpoint should be saved"""
        return (
            self.generation_config.checkpoint_frequency > 0 and
            current_idx > 0 and
            current_idx % self.generation_config.checkpoint_frequency == 0
        )


class SimplePipeline(BasePipeline):
    """Simple sequential generation pipeline"""
    
    def run(self) -> Dict[str, Any]:
        """Run the generation pipeline"""
        logger.info("Starting simple pipeline")
        
        # Load data
        df = self.data_loader.load_subset(
            start_index=self.generation_config.start_index,
            end_index=self.generation_config.end_index,
            max_samples=self.generation_config.max_samples
        )
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Process data
        items = self.data_processor.process_batch(
            df,
            question_column=self.data_loader.question_column
        )
        
        # Check for checkpoint
        checkpoint = self._load_checkpoint()
        start_idx = 0
        if checkpoint:
            start_idx = checkpoint.get("completed_items", 0)
            items = items[start_idx:]
            logger.info(f"Resuming from item {start_idx}")
        
        # Generate responses
        if self.generation_config.num_repeats > 1:
            results = self.generation_manager.generate_with_repeats(
                items,
                num_repeats=self.generation_config.num_repeats,
                repeat_strategy=self.generation_config.repeat_strategy,
                temperature_schedule=self.generation_config.temperature_schedule,
                repeat_order=self.generation_config.repeat_order,
                progress_callback=self._progress_callback
            )
        else:
            results = self.generation_manager.generate_batch(
                items,
                progress_callback=self._progress_callback
            )
        
        # Adjust indices if resuming
        if start_idx > 0:
            for result in results:
                result["idx"] += start_idx
        
        # Write results
        output_path = self.tracker.get_output_path()
        output_df = self.data_writer.write_results(
            df,
            results,
            output_path,
            num_repeats=self.generation_config.num_repeats
        )
        
        # Aggregate if requested
        if self.generation_config.aggregate_responses:
            output_df = self.data_writer.aggregate_responses(
                output_df,
                method=self.generation_config.aggregation_method
            )
            # Save aggregated version
            agg_path = output_path.replace(".parquet", "_aggregated.parquet")
            output_df.to_parquet(agg_path)
        
        # Get final metrics
        metrics = self.generation_manager.get_metrics()
        
        # Save metadata
        if self.data_writer.save_metadata:
            metadata = {
                "model_config": self.model_config.to_dict(),
                "generation_config": self.generation_config.__dict__,
                "metrics": metrics,
                "total_samples": len(df),
                "output_path": str(output_path)
            }
            self.data_writer.write_metadata(
                metadata,
                self.tracker.get_metadata_path()
            )
        
        logger.info("Pipeline completed successfully")
        return {
            "output_path": output_path,
            "metrics": metrics,
            "total_samples": len(df)
        }
    
    def _progress_callback(self, completed: int, total: int, progress_type: str = "items"):
        """Progress callback for tracking
        
        Args:
            completed: Number of completed units
            total: Total number of units
            progress_type: Type of progress ("items", "repeats")
        """
        if self.tracker:
            self.tracker.update_progress(completed, total, progress_type)
        
        # Save checkpoint if needed (only for item progress)
        if progress_type == "items" and self._should_save_checkpoint(completed):
            checkpoint_data = {
                "completed_items": completed,
                "total_items": total,
                "metrics": self.generation_manager.get_metrics(),
                "repeat_order": self.generation_config.repeat_order
            }
            self._save_checkpoint(checkpoint_data)