"""Configuration schemas using Pydantic."""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class DataConfig(BaseModel):
    """Configuration for data input/output."""
    
    input_path: Path = Field(..., description="Path to input parquet file")
    output_path: Path = Field(..., description="Path to output parquet file")
    input_column: str = Field("question", description="Column name containing input text")
    output_column: str = Field("response", description="Column name for output text")
    copy_columns: Optional[List[str]] = Field(None, description="Additional columns to copy")
    filter_condition: Optional[str] = Field(None, description="Pandas query string to filter rows")
    limit: Optional[int] = Field(None, description="Limit number of rows to process")
    shuffle: bool = Field(False, description="Shuffle data before processing")
    
    @field_validator("input_path")
    def validate_input_path(cls, v: Path) -> Path:
        if not v.suffix == ".parquet":
            raise ValueError("Input file must be a parquet file")
        return v
    
    @field_validator("output_path")
    def validate_output_path(cls, v: Path) -> Path:
        if not v.suffix == ".parquet":
            raise ValueError("Output file must be a parquet file")
        return v


class ModelConfig(BaseModel):
    """Configuration for vLLM model endpoint."""
    
    url: HttpUrl = Field(..., description="vLLM server URL")
    api_key: Optional[str] = Field(None, description="API key if required")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional headers")


class GenerationConfig(BaseModel):
    """Configuration for text generation parameters."""
    
    num_samples: int = Field(1, ge=1, description="Number of samples per input")
    temperature: Union[float, List[float]] = Field(1.0, ge=0.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(-1, description="Top-k sampling (-1 for disabled)")
    max_tokens: int = Field(512, ge=1, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(None, description="Random seed")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    enable_thinking: bool = Field(True, description="Enable thinking mode for reasoning models")

    @field_validator("temperature")
    def validate_temperature(cls, v: Union[float, List[float]], info) -> Union[float, List[float]]:
        if isinstance(v, list):
            num_samples = info.data.get("num_samples", 1)
            if len(v) != num_samples:
                raise ValueError(f"Temperature list length ({len(v)}) must match num_samples ({num_samples})")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    
    split_id: Optional[int] = Field(None, ge=1, description="Split ID to process (1-indexed)")
    num_splits: Optional[int] = Field(None, ge=1, description="Total number of splits")
    online_saving: bool = Field(True, description="Save results after each batch")
    batch_size: int = Field(1, ge=1, description="Items per batch for online saving")
    temp_dir: str = Field("temp", description="Directory for batch files")
    cleanup_batches: bool = Field(True, description="Clean up batch files after merging")
    
    @model_validator(mode='after')
    def validate_splits(self) -> 'ProcessingConfig':
        """Validate split configuration."""
        if (self.split_id is None) != (self.num_splits is None):
            raise ValueError("Both split_id and num_splits must be specified together")
        if self.split_id is not None and self.split_id > self.num_splits:
            raise ValueError(f"split_id ({self.split_id}) cannot be greater than num_splits ({self.num_splits})")
        return self


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    
    max_retries: int = Field(3, ge=0, description="Maximum retries for failed requests")
    retry_delay: float = Field(1.0, ge=0, description="Delay between retries in seconds")
    timeout: float = Field(300.0, ge=1, description="Request timeout in seconds")
    backoff_factor: float = Field(2.0, ge=1, description="Exponential backoff factor")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field("INFO", description="Logging level")
    file: Optional[Path] = Field(None, description="Log file path")
    format: str = Field(
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format string"
    )
    rotation: Optional[str] = Field("1 day", description="Log rotation")
    retention: Optional[str] = Field("7 days", description="Log retention")
    
    @field_validator("level")
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class Config(BaseModel):
    """Main configuration schema."""
    
    data: DataConfig
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        super().model_post_init(__context)
        
        # Ensure log directory exists if log file is specified
        if self.logging.file:
            self.logging.file.parent.mkdir(parents=True, exist_ok=True)