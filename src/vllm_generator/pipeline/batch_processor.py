"""Batch processor for efficient data processing."""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
from tqdm.asyncio import tqdm

from ..config.schemas import Config
from ..data import DataLoader, DataWriter, DataProcessor
from ..models import GenerationManager
from ..utils import get_logger, get_timestamp


class CheckpointManager:
    """Manage checkpoints for resumable processing."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("CheckpointManager")
    
    def save_checkpoint(
        self,
        batch_id: int,
        processed_indices: List[int],
        results: Dict[str, Any]
    ) -> Path:
        """Save checkpoint for a batch."""
        checkpoint_data = {
            "batch_id": batch_id,
            "processed_indices": processed_indices,
            "results": results,
            "timestamp": get_timestamp()
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_batch_{batch_id:04d}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Saved checkpoint for batch {batch_id}")
        return checkpoint_file
    
    def load_checkpoint(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a batch."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_batch_{batch_id:04d}.json"
        
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        
        self.logger.debug(f"Loaded checkpoint for batch {batch_id}")
        return data
    
    def get_last_batch_id(self) -> int:
        """Get the last processed batch ID."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_batch_*.json"))
        
        if not checkpoint_files:
            return -1
        
        # Extract batch IDs from filenames
        batch_ids = []
        for file in checkpoint_files:
            try:
                batch_id = int(file.stem.split("_")[-1])
                batch_ids.append(batch_id)
            except ValueError:
                continue
        
        return max(batch_ids) if batch_ids else -1
    
    def cleanup(self) -> None:
        """Clean up checkpoint files."""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_batch_*.json"):
            checkpoint_file.unlink()
        self.logger.info("Cleaned up checkpoint files")


class BatchProcessor:
    """Process data in batches with checkpointing and progress tracking."""
    
    def __init__(self, config: Config):
        """Initialize batch processor."""
        self.config = config
        self.logger = get_logger("BatchProcessor")
        
        # Initialize components
        self.data_loader = DataLoader(config.data)
        self.data_writer = DataWriter(config.data)
        self.data_processor = DataProcessor()
        self.checkpoint_manager = CheckpointManager(config.processing.checkpoint_dir)
    
    async def process(
        self,
        generation_manager: GenerationManager,
        dry_run: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Process data with generation."""
        start_time = time.time()
        
        # Load data
        self.logger.info("Loading input data...")
        df = self.data_loader.load()
        total_rows = len(df)
        
        # Check for resume
        start_batch = 0
        if self.config.processing.resume:
            last_batch_id = self.checkpoint_manager.get_last_batch_id()
            if last_batch_id >= 0:
                start_batch = last_batch_id + 1
                self.logger.info(f"Resuming from batch {start_batch}")
        
        # Process in batches
        batch_size = self.config.processing.batch_size
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        processed_results = []
        batch_files = []
        
        # Setup progress bar
        pbar = None
        if progress_bar:
            pbar = tqdm(
                total=total_rows,
                desc="Processing",
                unit="prompts",
                initial=start_batch * batch_size
            )
        
        try:
            for batch_id in range(start_batch, total_batches):
                batch_start = batch_id * batch_size
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = df.iloc[batch_start:batch_end]
                
                self.logger.info(
                    f"Processing batch {batch_id + 1}/{total_batches} "
                    f"({batch_end - batch_start} items)"
                )
                
                # Process batch
                if dry_run:
                    # Simulate processing
                    await asyncio.sleep(0.1)
                    results = [{"text": f"Dry run response {i}"} for i in range(len(batch_df))]
                else:
                    # Get prompts
                    prompts = self.data_loader.get_input_texts(batch_df)
                    
                    # Generate responses
                    responses = await generation_manager.generate_batch(
                        prompts,
                        progress_callback=pbar.update if pbar else None
                    )
                    
                    # Extract texts
                    results = generation_manager.extract_texts_from_responses(responses)
                
                # Create output dataframe
                output_df = self.data_processor.create_response_dataframe(
                    batch_df,
                    results,
                    self.config.data.input_column,
                    self.config.data.output_column,
                    self.config.data.copy_columns
                )
                
                # Save batch
                batch_file = self.data_writer.write_batch(output_df, batch_id)
                batch_files.append(batch_file)
                
                # Save checkpoint
                if (batch_id + 1) % self.config.processing.checkpoint_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        batch_id,
                        list(range(batch_start, batch_end)),
                        {"batch_file": str(batch_file)}
                    )
                
                processed_results.extend(results)
                
                if pbar and not dry_run:
                    # Update progress if not already updated by callback
                    current = pbar.n
                    expected = batch_end
                    if current < expected:
                        pbar.update(expected - current)
        
        finally:
            if pbar:
                pbar.close()
        
        # Merge batch files
        self.logger.info("Merging batch files...")
        self.data_writer.merge_batch_files(batch_files)
        
        # Clean up checkpoints
        if not self.config.processing.resume:
            self.checkpoint_manager.cleanup()
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        stats = {
            "total_processed": len(processed_results),
            "processing_time": elapsed_time,
            "prompts_per_second": len(processed_results) / elapsed_time,
            "model_statistics": generation_manager.get_statistics()
        }
        
        self.logger.info(
            f"Processing completed: {stats['total_processed']} items in "
            f"{elapsed_time:.2f}s ({stats['prompts_per_second']:.2f} prompts/s)"
        )
        
        return stats