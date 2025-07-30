"""Tests for repeated sampling and order preservation."""

import pandas as pd
from pathlib import Path

from vllm_generator.data import DataProcessor
from vllm_generator.models import GenerationManager
from vllm_generator.config import Config, DataConfig, ModelConfig, GenerationConfig


class TestRepeatSamplingOrder:
    """Test that repeated sampling preserves order correctly."""
    
    def test_group_responses_preserves_order(self):
        """Test that grouping responses preserves input order."""
        processor = DataProcessor()
        
        # Simulate 3 samples each for 4 inputs
        responses = [
            "Q1_S1", "Q1_S2", "Q1_S3",
            "Q2_S1", "Q2_S2", "Q2_S3",
            "Q3_S1", "Q3_S2", "Q3_S3",
            "Q4_S1", "Q4_S2", "Q4_S3",
        ]
        
        grouped = processor.group_responses_by_input(responses, num_samples=3)
        
        assert len(grouped) == 4
        assert grouped[0] == ["Q1_S1", "Q1_S2", "Q1_S3"]
        assert grouped[1] == ["Q2_S1", "Q2_S2", "Q2_S3"]
        assert grouped[2] == ["Q3_S1", "Q3_S2", "Q3_S3"]
        assert grouped[3] == ["Q4_S1", "Q4_S2", "Q4_S3"]
    
    def test_create_response_dataframe_multi_sample_order(self):
        """Test that multi-sample dataframe creation preserves order."""
        processor = DataProcessor()
        
        input_df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "id": [1, 2, 3],
            "metadata": ["A", "B", "C"]
        })
        
        # 2 samples per question
        responses = [
            ["Q1_R1", "Q1_R2"],
            ["Q2_R1", "Q2_R2"],
            ["Q3_R1", "Q3_R2"]
        ]
        
        output_df = processor.create_response_dataframe(
            input_df,
            responses,
            "question",
            "response"
        )
        
        # Should have 6 rows (3 questions * 2 samples)
        assert len(output_df) == 6
        
        # Check order is preserved
        expected_responses = ["Q1_R1", "Q1_R2", "Q2_R1", "Q2_R2", "Q3_R1", "Q3_R2"]
        assert output_df["response"].tolist() == expected_responses
        
        # Check metadata is duplicated correctly
        expected_questions = ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"]
        assert output_df["question"].tolist() == expected_questions
        
        expected_ids = [1, 1, 2, 2, 3, 3]
        assert output_df["id"].tolist() == expected_ids
        
        # Check sample indices
        expected_sample_idx = [0, 1, 0, 1, 0, 1]
        assert output_df["sample_idx"].tolist() == expected_sample_idx
    
    def test_temperature_schedule_order(self):
        """Test that temperature schedule is applied in correct order."""
        config = Config(
            data=DataConfig(
                input_path=Path("input.parquet"),
                output_path=Path("output.parquet")
            ),
            models=[ModelConfig(url="http://localhost:8000")],
            generation=GenerationConfig(
                num_samples=4,
                temperature=[0.5, 0.7, 0.9, 1.1]
            )
        )
        
        manager = GenerationManager(config)
        client = list(manager.clients.values())[0]
        
        # Test that each sample gets the right temperature
        for i in range(4):
            request = client._prepare_request("Test", sample_idx=i)
            expected_temp = [0.5, 0.7, 0.9, 1.1][i]
            assert request["temperature"] == expected_temp
    
    def test_extract_texts_multi_sample_order(self):
        """Test extracting texts preserves sample order."""
        config = Config(
            data=DataConfig(
                input_path=Path("input.parquet"),
                output_path=Path("output.parquet")
            ),
            models=[ModelConfig(url="http://localhost:8000")],
            generation=GenerationConfig(num_samples=3)
        )
        
        manager = GenerationManager(config)
        
        # Mock responses with 3 samples each
        responses = [
            {
                "choices": [
                    {"text": "Q1_S1"},
                    {"text": "Q1_S2"},
                    {"text": "Q1_S3"}
                ]
            },
            {
                "choices": [
                    {"text": "Q2_S1"},
                    {"text": "Q2_S2"},
                    {"text": "Q2_S3"}
                ]
            }
        ]
        
        texts = manager.extract_texts_from_responses(responses)
        
        assert len(texts) == 2
        assert texts[0] == ["Q1_S1", "Q1_S2", "Q1_S3"]
        assert texts[1] == ["Q2_S1", "Q2_S2", "Q2_S3"]
    
    def test_batch_processing_order(self):
        """Test that batch processing maintains order across batches."""
        processor = DataProcessor()
        
        # Simulate processing 10 items in batches of 3
        all_responses = []
        batch_sizes = [3, 3, 3, 1]  # 4 batches
        
        for batch_idx, batch_size in enumerate(batch_sizes):
            batch_responses = [f"B{batch_idx}_R{i}" for i in range(batch_size)]
            all_responses.extend(batch_responses)
        
        # Group by 2 samples per input
        grouped = processor.group_responses_by_input(all_responses, num_samples=2)
        
        expected = [
            ["B0_R0", "B0_R1"],  # First input
            ["B0_R2", "B1_R0"],  # Second input (spans batches)
            ["B1_R1", "B1_R2"],  # Third input
            ["B2_R0", "B2_R1"],  # Fourth input
            ["B2_R2", "B3_R0"],  # Fifth input (spans batches)
        ]
        
        assert grouped == expected