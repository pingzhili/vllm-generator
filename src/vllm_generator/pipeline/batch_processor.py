"""Batch processor for efficient data processing."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
from tqdm import tqdm

from ..config.schemas import Config
from ..data import DataLoader, DataWriter, DataProcessor
from ..models import VLLMClient
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
        self.data_loader = DataLoader(config.data, config.processing)
        self.data_writer = DataWriter(config.data, config.processing)
        self.data_processor = DataProcessor()
        self.checkpoint_manager = CheckpointManager(config.processing.checkpoint_dir)
    
    def process(
        self,
        vllm_client: VLLMClient,
        dry_run: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Process data with generation."""
        start_time = time.time()
        
        # Load data (with split support)
        self.logger.info("Loading input data...")
        df = self.data_loader.load_split()
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
                    time.sleep(0.1)
                    results = [f"Dry run response {i}" for i in range(len(batch_df))]
                else:
                    # Get prompts
                    prompts = self.data_loader.get_input_texts(batch_df)
                    
                    # Generate responses based on num_samples
                    results = []
                    if self.config.generation.num_samples == 1:
                        # Single sample per prompt
                        for prompt in prompts:
                            try:
                                response = vllm_client.generate(prompt)
                                text = response.get("choices", [{}])[0].get("text", "")
                                results.append(text)
                            except Exception as e:
                                self.logger.error(f"Generation failed: {e}")
                                results.append("")
                            if pbar:
                                pbar.update(1)
                    else:
                        # Multiple samples per prompt
                        for prompt in prompts:
                            prompt_results = []
                            for sample_idx in range(self.config.generation.num_samples):
                                try:
                                    response = vllm_client.generate(prompt, sample_idx=sample_idx)
                                    text = response.get("choices", [{}])[0].get("text", "")
                                    prompt_results.append(text)
                                except Exception as e:
                                    self.logger.error(f"Generation failed for sample {sample_idx}: {e}")
                                    prompt_results.append("")
                            results.append(prompt_results)
                            if pbar:
                                pbar.update(1)
                
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
                
                # Progress bar is already updated in the generation loop
        
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
        
        # Count total responses (handling both single and multiple samples)
        total_responses = 0
        if processed_results and isinstance(processed_results[0], list):
            # Multiple samples per prompt
            total_responses = sum(len(r) for r in processed_results)
        else:
            # Single sample per prompt
            total_responses = len(processed_results)
        
        stats = {
            "total_prompts": len(processed_results),
            "total_responses": total_responses,
            "processing_time": elapsed_time,
            "prompts_per_second": len(processed_results) / elapsed_time if elapsed_time > 0 else 0,
            "responses_per_second": total_responses / elapsed_time if elapsed_time > 0 else 0
        }
        
        self.logger.info(
            f"Processing completed: {stats['total_prompts']} prompts, "
            f"{stats['total_responses']} responses in {elapsed_time:.2f}s "
            f"({stats['prompts_per_second']:.2f} prompts/s)"
        )
        
        return stats