"""Data loader for parquet files."""

from typing import List, Iterator
import pandas as pd
import pyarrow.parquet as pq

from ..config.schemas import DataConfig
from ..utils import get_logger, chunk_dataframe


class DataLoader:
    """Load and process data from parquet files."""
    
    def __init__(self, config: DataConfig):
        """Initialize data loader with configuration."""
        self.config = config
        self.logger = get_logger("DataLoader")
        
    def load(self) -> pd.DataFrame:
        """Load data from parquet file."""
        self.logger.info(f"Loading data from {self.config.input_path}")
        
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")
        
        # Load parquet file
        df = pd.read_parquet(self.config.input_path)
        self.logger.info(f"Loaded {len(df)} rows from {self.config.input_path}")
        
        # Validate required columns
        if self.config.input_column not in df.columns:
            raise ValueError(f"Input column '{self.config.input_column}' not found in dataframe")
        
        # Apply filter if specified
        if self.config.filter_condition:
            original_len = len(df)
            df = df.query(self.config.filter_condition)
            self.logger.info(f"Filtered from {original_len} to {len(df)} rows")
        
        # Apply limit if specified
        if self.config.limit:
            df = df.head(self.config.limit)
            self.logger.info(f"Limited to {len(df)} rows")
        
        # Shuffle if requested
        if self.config.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Data shuffled")
        
        return df
    
    def load_schema(self) -> dict:
        """Load parquet schema without loading data."""
        schema = pq.read_schema(self.config.input_path)
        return {field.name: str(field.type) for field in schema}
    
    def load_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Load data in chunks for memory efficiency."""
        self.logger.info(f"Loading data in chunks of {chunk_size}")
        
        # First load full dataframe to apply filters
        df = self.load()
        
        # Yield chunks
        for i, chunk in enumerate(chunk_dataframe(df, chunk_size)):
            self.logger.debug(f"Yielding chunk {i+1} with {len(chunk)} rows")
            yield chunk
    
    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing_columns = []
        
        # Check input column
        if self.config.input_column not in df.columns:
            missing_columns.append(self.config.input_column)
        
        # Check copy columns if specified
        if self.config.copy_columns:
            for col in self.config.copy_columns:
                if col not in df.columns:
                    missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing columns in dataframe: {missing_columns}")
    
    def get_input_texts(self, df: pd.DataFrame) -> List[str]:
        """Extract input texts from dataframe."""
        return df[self.config.input_column].astype(str).tolist()
    
    def prepare_output_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare output dataframe with required columns."""
        # Start with input column
        output_df = df[[self.config.input_column]].copy()
        
        # Add copy columns if specified
        if self.config.copy_columns:
            for col in self.config.copy_columns:
                if col in df.columns:
                    output_df[col] = df[col]
        
        # Add placeholder for output column
        output_df[self.config.output_column] = None
        
        return output_df