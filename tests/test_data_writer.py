import pytest
import pandas as pd
import json
from pathlib import Path

from vllm_generator import DataWriter


class TestDataWriter:
    
    def test_write_results_wide_format(self, sample_dataframe, mock_generation_results, temp_dir):
        """Test writing results in wide format"""
        writer = DataWriter(output_format="wide")
        output_path = temp_dir / "output_wide.parquet"
        
        # Add repeat results
        results = mock_generation_results.copy()
        for i, result in enumerate(mock_generation_results):
            repeat_result = result.copy()
            repeat_result["repeat_id"] = 1
            repeat_result["response"] = f"{result['response']} (repeat)"
            results.append(repeat_result)
        
        output_df = writer.write_results(
            sample_dataframe.head(2),
            results,
            str(output_path),
            num_repeats=2
        )
        
        assert output_path.exists()
        assert len(output_df) == 2
        assert "response_0" in output_df.columns
        assert "response_1" in output_df.columns
        assert output_df.iloc[0]["response_0"] == "The capital of France is Paris."
        assert output_df.iloc[0]["response_1"] == "The capital of France is Paris. (repeat)"
    
    def test_write_results_long_format(self, sample_dataframe, mock_generation_results, temp_dir):
        """Test writing results in long format"""
        writer = DataWriter(output_format="long")
        output_path = temp_dir / "output_long.parquet"
        
        # Add repeat results
        results = []
        for repeat_id in range(2):
            for result in mock_generation_results:
                repeat_result = result.copy()
                repeat_result["repeat_id"] = repeat_id
                results.append(repeat_result)
        
        output_df = writer.write_results(
            sample_dataframe.head(2),
            results,
            str(output_path),
            num_repeats=2
        )
        
        assert output_path.exists()
        assert len(output_df) == 4  # 2 questions × 2 repeats
        assert "repeat_id" in output_df.columns
        assert "response" in output_df.columns
    
    def test_write_results_nested_format(self, sample_dataframe, mock_generation_results, temp_dir):
        """Test writing results in nested format"""
        writer = DataWriter(output_format="nested")
        output_path = temp_dir / "output_nested.parquet"
        
        # Add repeat results
        results = []
        for i, base_result in enumerate(mock_generation_results):
            for repeat_id in range(2):
                result = base_result.copy()
                result["repeat_id"] = repeat_id
                result["response"] = f"{base_result['response']} v{repeat_id}"
                results.append(result)
        
        output_df = writer.write_results(
            sample_dataframe.head(2),
            results,
            str(output_path),
            num_repeats=2
        )
        
        assert output_path.exists()
        assert len(output_df) == 2
        assert "response" in output_df.columns
        assert isinstance(output_df.iloc[0]["response"], list)
        assert len(output_df.iloc[0]["response"]) == 2
    
    def test_write_metadata(self, temp_dir):
        """Test writing metadata"""
        writer = DataWriter(save_metadata=True)
        metadata_path = temp_dir / "metadata.json"
        
        metadata = {
            "model": "gpt2",
            "temperature": 0.7,
            "total_samples": 100,
            "metrics": {"tokens_per_second": 150}
        }
        
        writer.write_metadata(metadata, str(metadata_path))
        
        assert metadata_path.exists()
        with open(metadata_path) as f:
            loaded = json.load(f)
        assert loaded["model"] == "gpt2"
        assert "timestamp" in loaded
    
    def test_checkpoint_operations(self, temp_dir):
        """Test checkpoint save and load"""
        writer = DataWriter()
        checkpoint_path = temp_dir / "checkpoint.json"
        
        checkpoint_data = {
            "completed_items": 50,
            "total_items": 100,
            "last_index": 49
        }
        
        writer.write_checkpoint(checkpoint_data, str(checkpoint_path))
        assert checkpoint_path.exists()
        
        loaded = DataWriter.load_checkpoint(str(checkpoint_path))
        assert loaded["completed_items"] == 50
        assert "checkpoint_time" in loaded
    
    def test_merge_results(self, sample_dataframe, temp_dir):
        """Test merging multiple result files"""
        writer = DataWriter()
        
        # Create multiple result files
        df1 = sample_dataframe.head(2).copy()
        df1["response"] = ["Answer 1", "Answer 2"]
        df2 = sample_dataframe.tail(3).copy()
        df2["response"] = ["Answer 3", "Answer 4", "Answer 5"]
        
        file1 = temp_dir / "results1.parquet"
        file2 = temp_dir / "results2.parquet"
        df1.to_parquet(file1)
        df2.to_parquet(file2)
        
        # Merge files
        output_path = temp_dir / "merged.parquet"
        merged_df = writer.merge_results(
            [str(file1), str(file2)],
            str(output_path),
            strategy="sequential"
        )
        
        assert output_path.exists()
        assert len(merged_df) == 5
        assert list(merged_df["response"]) == ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]
    
    def test_aggregate_responses(self, sample_dataframe):
        """Test response aggregation"""
        writer = DataWriter(output_column_prefix="response")
        
        # Create dataframe with multiple responses
        df = sample_dataframe.head(2).copy()
        df["response_0"] = ["Paris is the capital.", "Photosynthesis converts light."]
        df["response_1"] = ["The capital of France is Paris.", "Plants use photosynthesis."]
        df["response_2"] = ["Paris.", "Photosynthesis is a biological process."]
        
        # Test first aggregation
        result = writer.aggregate_responses(df.copy(), method="first")
        assert "response_aggregated" in result.columns
        assert result.iloc[0]["response_aggregated"] == "Paris is the capital."
        
        # Test longest aggregation
        result = writer.aggregate_responses(df.copy(), method="longest")
        assert result.iloc[0]["response_aggregated"] == "The capital of France is Paris."
        assert result.iloc[1]["response_aggregated"] == "Photosynthesis is a biological process."
        
        # Test majority vote
        df_vote = df.copy()
        df_vote["response_0"] = ["Paris", "Process A"]
        df_vote["response_1"] = ["Paris", "Process B"]
        df_vote["response_2"] = ["London", "Process A"]
        
        result = writer.aggregate_responses(df_vote, method="majority_vote")
        assert result.iloc[0]["response_aggregated"] == "Paris"
        assert result.iloc[1]["response_aggregated"] == "Process A"
    
    def test_save_metadata_in_results(self, sample_dataframe, mock_generation_results, temp_dir):
        """Test saving metadata with results"""
        writer = DataWriter(save_metadata=True)
        output_path = temp_dir / "output_with_meta.parquet"
        
        # Add metadata to results
        results = mock_generation_results.copy()
        for result in results:
            result["tokens"] = 10
            result["latency"] = 0.1
        
        metadata = {"generation_config": {"temperature": 0.7}}
        
        output_df = writer.write_results(
            sample_dataframe.head(2),
            results,
            str(output_path),
            metadata=metadata
        )
        
        assert output_path.exists()
        assert "response_0_tokens" in output_df.columns
        assert "response_0_latency" in output_df.columns