"""Data writer for saving results to parquet files."""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config.schemas import DataConfig
from ..utils import get_logger, ensure_directory


class DataWriter:
    """Write processed data to parquet files."""
    
    def __init__(self, config: DataConfig):
        """Initialize data writer with configuration."""
        self.config = config
        self.logger = get_logger("DataWriter")
        
        # Ensure output directory exists
        ensure_directory(self.config.output_path.parent)
    
    def write(self, df: pd.DataFrame) -> None:
        """Write dataframe to parquet file."""
        self.logger.info(f"Writing {len(df)} rows to {self.config.output_path}")
        
        # Write to parquet
        df.to_parquet(
            self.config.output_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        
        self.logger.info(f"Successfully wrote data to {self.config.output_path}")
    
    def write_batch(
        self,
        df: pd.DataFrame,
        batch_id: int,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Write a batch of data with unique filename."""
        if output_dir is None:
            output_dir = self.config.output_path.parent
        
        # Create batch filename
        batch_filename = f"{self.config.output_path.stem}_batch_{batch_id:04d}.parquet"
        batch_path = output_dir / batch_filename
        
        self.logger.debug(f"Writing batch {batch_id} to {batch_path}")
        
        df.to_parquet(
            batch_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        
        return batch_path
    
    def append_results(
        self,
        df: pd.DataFrame,
        responses: Union[List[str], List[List[str]]],
        indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Append responses to dataframe."""
        if indices is None:
            indices = list(range(len(responses)))
        
        # Handle single responses
        if responses and isinstance(responses[0], str):
            for idx, response in zip(indices, responses):
                df.loc[idx, self.config.output_column] = response
        
        # Handle multiple responses per input
        elif responses and isinstance(responses[0], list):
            # Create expanded dataframe
            expanded_rows = []
            
            for idx, response_list in zip(indices, responses):
                for response in response_list:
                    row = df.loc[idx].copy()
                    row[self.config.output_column] = response
                    expanded_rows.append(row)
            
            df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        return df
    
    def merge_batch_files(
        self,
        batch_files: List[Path],
        output_path: Optional[Path] = None
    ) -> None:
        """Merge multiple batch files into single output file."""
        if not batch_files:
            self.logger.warning("No batch files to merge")
            return
        
        if output_path is None:
            output_path = self.config.output_path
        
        self.logger.info(f"Merging {len(batch_files)} batch files")
        
        # Read and concatenate all batches
        dfs = []
        for batch_file in batch_files:
            if batch_file.exists():
                dfs.append(pd.read_parquet(batch_file))
        
        if not dfs:
            self.logger.warning("No valid batch files found")
            return
        
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Write merged result
        self.write(merged_df)
        
        # Clean up batch files
        for batch_file in batch_files:
            if batch_file.exists():
                batch_file.unlink()
                self.logger.debug(f"Deleted batch file: {batch_file}")
    
    def write_checkpoint(
        self,
        df: pd.DataFrame,
        checkpoint_id: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """Write checkpoint file with metadata."""
        checkpoint_dir = self.config.output_path.parent / "checkpoints"
        ensure_directory(checkpoint_dir)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_id}.parquet"
        
        # Add metadata to parquet file
        table = pa.Table.from_pandas(df)
        
        # Convert metadata values to strings
        metadata_str = {k: str(v) for k, v in metadata.items()}
        table = table.replace_schema_metadata(metadata_str)
        
        pq.write_table(table, checkpoint_path, compression="snappy")
        
        self.logger.debug(f"Wrote checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def read_checkpoint(self, checkpoint_path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Read checkpoint file with metadata."""
        table = pq.read_table(checkpoint_path)
        df = table.to_pandas()
        
        # Extract metadata
        metadata = {}
        if table.schema.metadata:
            metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
        
        return df, metadata