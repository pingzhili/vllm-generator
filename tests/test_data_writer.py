"""Tests for data writer."""

import pandas as pd
from pathlib import Path

from vllm_generator.data import DataWriter
from vllm_generator.config.schemas import DataConfig


class TestDataWriter:
    """Test data writer functionality."""
    
    def test_write_basic(self, temp_dir):
        """Test basic data writing."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "output.parquet",
            input_column="question",
            output_column="response"
        )
        
        writer = DataWriter(config)
        
        # Create test dataframe
        df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "response": ["R1", "R2", "R3"]
        })
        
        writer.write(df)
        
        # Verify file exists and content
        assert config.output_path.exists()
        loaded_df = pd.read_parquet(config.output_path)
        assert len(loaded_df) == 3
        assert loaded_df["question"].tolist() == ["Q1", "Q2", "Q3"]
        assert loaded_df["response"].tolist() == ["R1", "R2", "R3"]
    
    def test_write_batch(self, temp_dir):
        """Test writing batch files."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "output.parquet"
        )
        
        writer = DataWriter(config)
        
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "response": ["R1", "R2"]
        })
        
        batch_path = writer.write_batch(df, batch_id=1)
        
        assert batch_path.exists()
        assert "batch_0001" in str(batch_path)
        
        loaded_df = pd.read_parquet(batch_path)
        assert len(loaded_df) == 2
    
    def test_append_results_single(self, temp_dir):
        """Test appending single responses."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "output.parquet",
            output_column="response"
        )
        
        writer = DataWriter(config)
        
        df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "response": [None, None, None]
        })
        
        responses = ["R1", "R2", "R3"]
        result_df = writer.append_results(df, responses)
        
        assert result_df["response"].tolist() == ["R1", "R2", "R3"]
    
    def test_append_results_multiple(self, temp_dir):
        """Test appending multiple responses per input."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "output.parquet",
            output_column="response"
        )
        
        writer = DataWriter(config)
        
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "id": [1, 2]
        })
        
        responses = [["R1a", "R1b"], ["R2a", "R2b"]]
        result_df = writer.append_results(df, responses)
        
        # Should expand to 4 rows
        assert len(result_df) == 4
        assert result_df["response"].tolist() == ["R1a", "R1b", "R2a", "R2b"]
        assert result_df["question"].tolist() == ["Q1", "Q1", "Q2", "Q2"]
    
    def test_merge_batch_files(self, temp_dir):
        """Test merging multiple batch files."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "merged_output.parquet"
        )
        
        writer = DataWriter(config)
        
        # Create batch files
        batch_files = []
        for i in range(3):
            df = pd.DataFrame({
                "question": [f"Q{i}1", f"Q{i}2"],
                "response": [f"R{i}1", f"R{i}2"]
            })
            batch_path = writer.write_batch(df, batch_id=i)
            batch_files.append(batch_path)
        
        # Merge
        writer.merge_batch_files(batch_files)
        
        # Verify merged file
        assert config.output_path.exists()
        merged_df = pd.read_parquet(config.output_path)
        assert len(merged_df) == 6
        
        # Verify batch files deleted
        for batch_file in batch_files:
            assert not batch_file.exists()
    
    def test_checkpoint_operations(self, temp_dir):
        """Test checkpoint write and read."""
        config = DataConfig(
            input_path=Path("input.parquet"),
            output_path=temp_dir / "output.parquet"
        )
        
        writer = DataWriter(config)
        
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "response": ["R1", "R2"]
        })
        
        metadata = {
            "batch_id": 1,
            "timestamp": "2024-01-01",
            "processed": 100
        }
        
        # Write checkpoint
        checkpoint_path = writer.write_checkpoint(df, "test_checkpoint", metadata)
        assert checkpoint_path.exists()
        
        # Read checkpoint
        loaded_df, loaded_metadata = writer.read_checkpoint(checkpoint_path)
        
        assert len(loaded_df) == 2
        assert loaded_df["question"].tolist() == ["Q1", "Q2"]
        assert loaded_metadata["batch_id"] == "1"
        assert loaded_metadata["processed"] == "100"