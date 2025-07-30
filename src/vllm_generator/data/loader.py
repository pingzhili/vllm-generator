"""Data loader for parquet files with splitting support."""

from typing import List, Iterator, Tuple
import pandas as pd
import pyarrow.parquet as pq

from ..config.schemas import DataConfig, ProcessingConfig
from ..utils import get_logger, chunk_dataframe


class DataLoader:
    """Load and process data from parquet files with optional splitting."""
    
    def __init__(self, data_config: DataConfig, processing_config: ProcessingConfig):
        """Initialize data loader with configuration."""
        self.data_config = data_config
        self.processing_config = processing_config
        self.logger = get_logger("DataLoader")
        
    def load(self) -> pd.DataFrame:
        """Load data from parquet file."""
        self.logger.info(f"Loading data from {self.data_config.input_path}")
        
        if not self.data_config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.data_config.input_path}")
        
        # Load parquet file
        df = pd.read_parquet(self.data_config.input_path)
        self.logger.info(f"Loaded {len(df)} rows from {self.data_config.input_path}")
        
        # Validate required columns
        if self.data_config.input_column not in df.columns:
            raise ValueError(f"Input column '{self.data_config.input_column}' not found in dataframe")
        
        # Apply filter if specified
        if self.data_config.filter_condition:
            original_len = len(df)
            df = df.query(self.data_config.filter_condition)
            self.logger.info(f"Filtered from {original_len} to {len(df)} rows")
        
        # Apply limit if specified
        if self.data_config.limit:
            df = df.head(self.data_config.limit)
            self.logger.info(f"Limited to {len(df)} rows")
        
        # Shuffle if requested
        if self.data_config.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Data shuffled")
        
        return df
    
    def load_split(self) -> pd.DataFrame:
        """Load specific split of data based on split_id and num_splits."""
        # First load the full data with filters applied
        df = self.load()
        
        # If no split configuration, return full dataframe
        if self.processing_config.split_id is None or self.processing_config.num_splits is None:
            return df
        
        # Calculate split boundaries
        total_rows = len(df)
        split_id = self.processing_config.split_id
        num_splits = self.processing_config.num_splits
        
        # Calculate base split size and remainder
        base_split_size = total_rows // num_splits
        remainder = total_rows % num_splits
        
        # Calculate start and end indices for this split
        # Splits 1 to remainder get one extra row
        if split_id <= remainder:
            start_idx = (split_id - 1) * (base_split_size + 1)
            end_idx = start_idx + base_split_size + 1
        else:
            start_idx = remainder * (base_split_size + 1) + (split_id - remainder - 1) * base_split_size
            end_idx = start_idx + base_split_size
        
        # Extract the split
        split_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        self.logger.info(
            f"Processing split {split_id}/{num_splits}: "
            f"rows {start_idx}-{end_idx-1} ({len(split_df)} rows)"
        )
        
        return split_df
    
    def get_split_info(self) -> Tuple[int, int]:
        """Get split boundaries without loading data."""
        # Load just to get total count after filters
        df = self.load()
        total_rows = len(df)
        
        if self.processing_config.split_id is None:
            return 0, total_rows
        
        split_id = self.processing_config.split_id
        num_splits = self.processing_config.num_splits
        
        base_split_size = total_rows // num_splits
        remainder = total_rows % num_splits
        
        if split_id <= remainder:
            start_idx = (split_id - 1) * (base_split_size + 1)
            end_idx = start_idx + base_split_size + 1
        else:
            start_idx = remainder * (base_split_size + 1) + (split_id - remainder - 1) * base_split_size
            end_idx = start_idx + base_split_size
        
        return start_idx, end_idx
    
    def load_schema(self) -> dict:
        """Load parquet schema without loading data."""
        schema = pq.read_schema(self.data_config.input_path)
        return {field.name: str(field.type) for field in schema}
    
    def load_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Load data in chunks for memory efficiency."""
        self.logger.info(f"Loading data in chunks of {chunk_size}")
        
        # First load the split
        df = self.load_split()
        
        # Yield chunks
        for i, chunk in enumerate(chunk_dataframe(df, chunk_size)):
            self.logger.debug(f"Yielding chunk {i+1} with {len(chunk)} rows")
            yield chunk
    
    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing_columns = []
        
        # Check input column
        if self.data_config.input_column not in df.columns:
            missing_columns.append(self.data_config.input_column)
        
        # Check copy columns if specified
        if self.data_config.copy_columns:
            for col in self.data_config.copy_columns:
                if col not in df.columns:
                    missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing columns in dataframe: {missing_columns}")
    
    def get_input_texts(self, df: pd.DataFrame) -> List[str]:
        """Extract input texts from dataframe."""
        return df[self.data_config.input_column].astype(str).tolist()
    
    def prepare_output_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare output dataframe with required columns."""
        # Start with input column
        output_df = df[[self.data_config.input_column]].copy()
        
        # Add copy columns if specified
        if self.data_config.copy_columns:
            for col in self.data_config.copy_columns:
                if col in df.columns:
                    output_df[col] = df[col]
        
        # Add placeholder for output column
        output_df[self.data_config.output_column] = None
        
        return output_df