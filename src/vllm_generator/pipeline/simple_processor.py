"""Simple processor for one-by-one data processing."""

from typing import Dict, Any, List
import time
from tqdm import tqdm
import pandas as pd

from ..config.schemas import Config
from ..data import DataLoader, DataWriter
from ..models import VLLMClient
from ..utils import get_logger


class SimpleProcessor:
    """Process data one item at a time without batching."""
    
    def __init__(self, config: Config):
        """Initialize simple processor."""
        self.config = config
        self.logger = get_logger("SimpleProcessor")
        
        # Initialize components
        self.data_loader = DataLoader(config.data, config.processing)
        self.data_writer = DataWriter(config.data, config.processing)
    
    def process(
        self,
        vllm_client: VLLMClient,
        dry_run: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Process data one item at a time."""
        start_time = time.time()
        
        # Load data (with split support)
        self.logger.info("Loading input data...")
        df = self.data_loader.load_split()
        total_items = len(df)
        
        self.logger.info(f"Processing {total_items} items...")
        
        # Prepare results storage
        all_results = []
        
        # Setup progress bar
        pbar = None
        if progress_bar:
            pbar = tqdm(total=total_items, desc="Processing", unit="items")
        
        try:
            # Process each item
            for idx, row in df.iterrows():
                prompt = str(row[self.config.data.input_column])
                
                if dry_run:
                    # Simulate processing
                    time.sleep(0.01)
                    if self.config.generation.num_samples == 1:
                        results = [(f"Dry run response for: {prompt[:50]}...", "Dry run thinking...")]
                    else:
                        results = [(f"Dry run sample {i+1}", f"Dry run thinking {i+1}") for i in range(self.config.generation.num_samples)]
                else:
                    # Generate real responses
                    if self.config.generation.num_samples == 1:
                        # Single sample
                        try:
                            response = vllm_client.generate(prompt)
                            choice = response.get("choices", [{}])[0]
                            text = choice.get("text", "")
                            thinking_text = choice.get("thinking_text", "")
                            results = [(text, thinking_text)]
                        except Exception as e:
                            self.logger.error(f"Generation failed for item {idx}: {e}")
                            results = [("", "")]
                    else:
                        # Multiple samples
                        results = []
                        for sample_idx in range(self.config.generation.num_samples):
                            try:
                                response = vllm_client.generate(prompt, sample_idx=sample_idx)
                                choice = response.get("choices", [{}])[0]
                                text = choice.get("text", "")
                                thinking_text = choice.get("thinking_text", "")
                                results.append((text, thinking_text))
                            except Exception as e:
                                self.logger.error(f"Generation failed for item {idx}, sample {sample_idx}: {e}")
                                results.append(("", ""))
                
                # Store results
                if self.config.generation.num_samples == 1:
                    # Single response per input
                    result_row = row.to_dict()
                    response_text, thinking_text = results[0]
                    result_row[self.config.data.output_column] = response_text
                    result_row["thinking_text"] = thinking_text
                    all_results.append(result_row)
                else:
                    # Multiple responses per input
                    for sample_idx, (response_text, thinking_text) in enumerate(results):
                        result_row = row.to_dict()
                        result_row[self.config.data.output_column] = response_text
                        result_row["thinking_text"] = thinking_text
                        result_row["sample_idx"] = sample_idx
                        all_results.append(result_row)
                
                if pbar:
                    pbar.update(1)
        
        finally:
            if pbar:
                pbar.close()
        
        # Create output dataframe
        output_df = pd.DataFrame(all_results)
        
        # Save results
        self.logger.info("Saving results...")
        self.data_writer.write(output_df)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        total_responses = len(all_results)
        
        stats = {
            "total_items": total_items,
            "total_responses": total_responses,
            "processing_time": elapsed_time,
            "items_per_second": total_items / elapsed_time if elapsed_time > 0 else 0,
            "responses_per_second": total_responses / elapsed_time if elapsed_time > 0 else 0
        }
        
        self.logger.info(
            f"Processing completed: {stats['total_items']} items, "
            f"{stats['total_responses']} responses in {elapsed_time:.2f}s "
            f"({stats['items_per_second']:.2f} items/s)"
        )
        
        return stats