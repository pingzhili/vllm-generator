"""Tests for data loader."""

import pytest
import pandas as pd
from pathlib import Path

from vllm_generator.data import DataLoader
from vllm_generator.config.schemas import DataConfig


class TestDataLoader:
    """Test data loader functionality."""
    
    def test_load_basic(self, sample_parquet_file):
        """Test basic data loading."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            input_column="question",
            output_column="response"
        )
        
        loader = DataLoader(config)
        df = loader.load()
        
        assert len(df) == 5
        assert "question" in df.columns
        assert df["question"].iloc[0] == "What is the capital of France?"
    
    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file."""
        config = DataConfig(
            input_path=temp_dir / "nonexistent.parquet",
            output_path=Path("output.parquet")
        )
        
        loader = DataLoader(config)
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_load_with_filter(self, sample_parquet_file):
        """Test loading with filter condition."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            filter_condition="category == 'ai'"
        )
        
        loader = DataLoader(config)
        df = loader.load()
        
        assert len(df) == 2
        assert all(df["category"] == "ai")
    
    def test_load_with_limit(self, sample_parquet_file):
        """Test loading with row limit."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            limit=3
        )
        
        loader = DataLoader(config)
        df = loader.load()
        
        assert len(df) == 3
    
    def test_load_with_shuffle(self, sample_parquet_file):
        """Test loading with shuffle."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            shuffle=True
        )
        
        loader = DataLoader(config)
        df1 = loader.load()
        df2 = loader.load()
        
        # With shuffle, order should be different (with high probability)
        # We can't guarantee they're different, but we can check they're valid
        assert len(df1) == 5
        assert len(df2) == 5
        assert set(df1["id"].values) == set(df2["id"].values)
    
    def test_validate_columns(self, sample_parquet_file):
        """Test column validation."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            input_column="question",
            copy_columns=["id", "category"]
        )
        
        loader = DataLoader(config)
        df = loader.load()
        
        # Should pass
        loader.validate_columns(df)
        
        # Test with missing column
        config.input_column = "missing_column"
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="Missing columns"):
            loader.validate_columns(df)
    
    def test_load_chunks(self, sample_parquet_file):
        """Test loading data in chunks."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet")
        )
        
        loader = DataLoader(config)
        chunks = list(loader.load_chunks(chunk_size=2))
        
        assert len(chunks) == 3  # 5 rows with chunk size 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1
    
    def test_get_input_texts(self, sample_parquet_file):
        """Test extracting input texts."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            input_column="question"
        )
        
        loader = DataLoader(config)
        df = loader.load()
        texts = loader.get_input_texts(df)
        
        assert len(texts) == 5
        assert texts[0] == "What is the capital of France?"
        assert all(isinstance(text, str) for text in texts)
    
    def test_prepare_output_dataframe(self, sample_parquet_file):
        """Test preparing output dataframe."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet"),
            input_column="question",
            output_column="response",
            copy_columns=["id", "category"]
        )
        
        loader = DataLoader(config)
        df = loader.load()
        output_df = loader.prepare_output_dataframe(df)
        
        assert "question" in output_df.columns
        assert "response" in output_df.columns
        assert "id" in output_df.columns
        assert "category" in output_df.columns
        assert output_df["response"].isna().all()  # Should be None initially
    
    def test_load_schema(self, sample_parquet_file):
        """Test loading parquet schema."""
        config = DataConfig(
            input_path=sample_parquet_file,
            output_path=Path("output.parquet")
        )
        
        loader = DataLoader(config)
        schema = loader.load_schema()
        
        assert "question" in schema
        assert "id" in schema
        assert "category" in schema
        assert "difficulty" in schema