import pytest
import pandas as pd
from pathlib import Path

from vllm_generator import ModelConfig, GenerationConfig
from vllm_generator import DataLoader, DataProcessor, DataWriter
from vllm_generator import SimplePipeline
from vllm_generator import GenerationTracker


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.mark.integration
    def test_simple_pipeline_end_to_end(self, sample_parquet_file, temp_dir):
        """Test complete pipeline execution"""
        # Create configurations
        model_config = ModelConfig(model="gpt2", max_tokens=50)
        generation_config = GenerationConfig(
            batch_size=2,
            num_repeats=1,
            checkpoint_frequency=0  # Disable for test
        )
        
        # Create components
        data_loader = DataLoader(sample_parquet_file, question_column="question")
        data_processor = DataProcessor(
            prompt_template="Question: {question}\nAnswer:"
        )
        data_writer = DataWriter(output_format="wide")
        tracker = GenerationTracker(output_dir=temp_dir, enable_progress_bar=False)
        
        # Create pipeline
        pipeline = SimplePipeline(
            model_config=model_config,
            generation_config=generation_config,
            data_loader=data_loader,
            data_processor=data_processor,
            data_writer=data_writer,
            tracker=tracker
        )
        
        # Initialize and run
        pipeline.initialize()
        try:
            result = pipeline.run()
            
            # Check results
            assert "output_path" in result
            assert "metrics" in result
            assert result["total_samples"] == 5
            
            # Check output file exists
            output_path = Path(result["output_path"])
            assert output_path.exists()
            
            # Load and check output
            output_df = pd.read_parquet(output_path)
            assert len(output_df) == 5
            assert "response_0" in output_df.columns
            
            # Check all questions got responses
            for idx in range(len(output_df)):
                assert pd.notna(output_df.iloc[idx]["response_0"])
        finally:
            pipeline.shutdown()
    
    @pytest.mark.integration
    def test_pipeline_with_repeats(self, sample_parquet_file, temp_dir):
        """Test pipeline with repeat generation"""
        model_config = ModelConfig(model="gpt2", max_tokens=30)
        generation_config = GenerationConfig(
            batch_size=2,
            num_repeats=3,
            repeat_strategy="temperature_schedule",
            temperature_schedule=[0.5, 1.0, 1.5]
        )
        
        # Create components
        data_loader = DataLoader(sample_parquet_file)
        data_processor = DataProcessor()
        data_writer = DataWriter(output_format="wide")
        tracker = GenerationTracker(output_dir=temp_dir, enable_progress_bar=False)
        
        # Create and run pipeline
        pipeline = SimplePipeline(
            model_config=model_config,
            generation_config=generation_config,
            data_loader=data_loader,
            data_processor=data_processor,
            data_writer=data_writer,
            tracker=tracker
        )
        
        pipeline.initialize()
        try:
            result = pipeline.run()
            
            # Check output has multiple response columns
            output_df = pd.read_parquet(result["output_path"])
            assert "response_0" in output_df.columns
            assert "response_1" in output_df.columns
            assert "response_2" in output_df.columns
        finally:
            pipeline.shutdown()
    
    @pytest.mark.integration
    def test_pipeline_with_checkpoint(self, sample_parquet_file, temp_dir):
        """Test pipeline checkpoint and resume"""
        model_config = ModelConfig(model="gpt2")
        generation_config = GenerationConfig(
            batch_size=1,
            checkpoint_frequency=2,
            max_samples=3  # Process only 3 samples
        )
        
        # Create components
        data_loader = DataLoader(sample_parquet_file)
        data_processor = DataProcessor()
        data_writer = DataWriter()
        tracker = GenerationTracker(output_dir=temp_dir, enable_progress_bar=False)
        
        # First run - should create checkpoint after 2 items
        pipeline = SimplePipeline(
            model_config=model_config,
            generation_config=generation_config,
            data_loader=data_loader,
            data_processor=data_processor,
            data_writer=data_writer,
            tracker=tracker
        )
        
        pipeline.initialize()
        try:
            _ = pipeline.run()
            
            # Check checkpoint was created
            checkpoint_path = Path(tracker.get_checkpoint_path())
            assert checkpoint_path.exists()
        finally:
            pipeline.shutdown()
    
    @pytest.mark.integration
    def test_pipeline_error_handling(self, sample_parquet_file, temp_dir):
        """Test pipeline error handling"""
        # Create a model config that might cause issues
        model_config = ModelConfig(model="gpt2", max_tokens=10)
        generation_config = GenerationConfig(
            batch_size=2,
            error_handling="skip",  # Skip errors
            max_retries=1
        )
        
        # Create components
        data_loader = DataLoader(sample_parquet_file)
        data_processor = DataProcessor()
        data_writer = DataWriter()
        tracker = GenerationTracker(output_dir=temp_dir, enable_progress_bar=False)
        
        pipeline = SimplePipeline(
            model_config=model_config,
            generation_config=generation_config,
            data_loader=data_loader,
            data_processor=data_processor,
            data_writer=data_writer,
            tracker=tracker
        )
        
        pipeline.initialize()
        try:
            # Should complete even if some generations fail
            result = pipeline.run()
            assert result["total_samples"] == 5
        finally:
            pipeline.shutdown()
    
    @pytest.mark.integration
    def test_pipeline_with_metadata(self, sample_parquet_file, temp_dir):
        """Test pipeline with metadata saving"""
        model_config = ModelConfig(model="gpt2")
        generation_config = GenerationConfig(batch_size=2)
        
        # Create components with metadata enabled
        data_loader = DataLoader(sample_parquet_file)
        data_processor = DataProcessor()
        data_writer = DataWriter(save_metadata=True)
        tracker = GenerationTracker(output_dir=temp_dir, enable_progress_bar=False)
        
        pipeline = SimplePipeline(
            model_config=model_config,
            generation_config=generation_config,
            data_loader=data_loader,
            data_processor=data_processor,
            data_writer=data_writer,
            tracker=tracker
        )
        
        pipeline.initialize()
        try:
            _ = pipeline.run()
            
            # Check metadata file was created
            metadata_path = Path(tracker.get_metadata_path())
            assert metadata_path.exists()
            
            # Load and check metadata
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert "model_config" in metadata
            assert "generation_config" in metadata
            assert "metrics" in metadata
            assert metadata["model_config"]["model"] == "gpt2"
        finally:
            pipeline.shutdown()