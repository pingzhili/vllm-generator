import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import DataLoader


class TestDataLoader:
    
    def test_load_parquet(self, sample_parquet_file):
        """Test loading a parquet file"""
        loader = DataLoader(sample_parquet_file, question_column="question")
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "question" in df.columns
        assert loader.total_rows == 5
    
    def test_validate_columns(self, sample_parquet_file):
        """Test column validation"""
        # Valid column
        loader = DataLoader(sample_parquet_file, question_column="question")
        df = loader.load()
        assert df is not None
        
        # Invalid column
        loader = DataLoader(sample_parquet_file, question_column="invalid_column")
        with pytest.raises(ValueError, match="Question column 'invalid_column' not found"):
            loader.load()
    
    def test_load_subset(self, sample_parquet_file):
        """Test loading subset of data"""
        loader = DataLoader(sample_parquet_file)
        
        # Test with max_samples
        df = loader.load_subset(max_samples=3)
        assert len(df) == 3
        
        # Test with start/end index
        df = loader.load_subset(start_index=1, end_index=4)
        assert len(df) == 3
        
        # Test with filter
        df = loader.load_subset(filter_column="category", filter_value="AI")
        assert len(df) == 1
        assert df.iloc[0]["category"] == "AI"
    
    def test_create_shards(self, sample_parquet_file, temp_dir):
        """Test creating data shards"""
        loader = DataLoader(sample_parquet_file)
        
        # Test contiguous sharding
        shards = loader.create_shards(num_shards=2, strategy="contiguous")
        assert len(shards) == 2
        assert shards[0]["num_rows"] + shards[1]["num_rows"] == 5
        
        # Test with output directory
        shard_dir = temp_dir / "shards"
        shards = loader.create_shards(
            num_shards=2, 
            output_dir=str(shard_dir),
            strategy="contiguous"
        )
        assert len(shards) == 2
        assert Path(shards[0]["path"]).exists()
        assert Path(shards[1]["path"]).exists()
    
    def test_estimate_tokens(self, sample_dataframe):
        """Test token estimation"""
        loader = DataLoader("dummy.parquet")
        tokens = loader.estimate_tokens(sample_dataframe)
        
        assert len(tokens) == len(sample_dataframe)
        assert all(t > 0 for t in tokens)
    
    def test_merge_shards(self, temp_dir, sample_dataframe):
        """Test merging shard files"""
        # Create shard files
        shard1_path = temp_dir / "shard1.parquet"
        shard2_path = temp_dir / "shard2.parquet"
        
        sample_dataframe.iloc[:3].to_parquet(shard1_path)
        sample_dataframe.iloc[3:].to_parquet(shard2_path)
        
        # Merge shards
        output_path = temp_dir / "merged.parquet"
        merged_df = DataLoader.merge_shards(
            [str(shard1_path), str(shard2_path)],
            str(output_path)
        )
        
        assert len(merged_df) == len(sample_dataframe)
        assert output_path.exists()
    
    def test_round_robin_sharding(self, sample_parquet_file):
        """Test round-robin sharding strategy"""
        loader = DataLoader(sample_parquet_file)
        shards = loader.create_shards(num_shards=2, strategy="round_robin")
        
        assert len(shards) == 2
        # Round-robin should distribute evenly
        assert abs(shards[0]["num_rows"] - shards[1]["num_rows"]) <= 1
    
    def test_balanced_sharding(self, sample_parquet_file):
        """Test balanced sharding strategy"""
        loader = DataLoader(sample_parquet_file)
        shards = loader.create_shards(num_shards=2, strategy="balanced")
        
        assert len(shards) == 2
        # Check that all rows are accounted for
        total_rows = sum(s["num_rows"] for s in shards)
        assert total_rows == 5