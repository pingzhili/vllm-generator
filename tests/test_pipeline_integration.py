"""Integration tests for the pipeline."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, patch

from vllm_generator.pipeline import GenerationPipeline, BatchProcessor
from vllm_generator.config import Config, DataConfig, ModelConfig
from vllm_generator.models import GenerationManager


class TestBatchProcessor:
    """Test BatchProcessor functionality."""
    
    @pytest.fixture
    def test_config(self, temp_dir, sample_parquet_file):
        """Create test configuration."""
        return Config(
            data=DataConfig(
                input_path=sample_parquet_file,
                output_path=temp_dir / "output.parquet",
                input_column="question",
                output_column="response"
            ),
            models=[ModelConfig(url="http://localhost:8000")]
        )
    
    @pytest.mark.asyncio
    async def test_process_dry_run(self, test_config):
        """Test batch processor in dry run mode."""
        processor = BatchProcessor(test_config)
        
        # Create mock generation manager
        gen_manager = AsyncMock(spec=GenerationManager)
        gen_manager.get_statistics.return_value = {"total_requests": 5}
        
        # Process in dry run mode
        stats = await processor.process(
            gen_manager,
            dry_run=True,
            progress_bar=False
        )
        
        assert stats["total_processed"] == 5
        assert stats["processing_time"] > 0
        assert stats["prompts_per_second"] > 0
        
        # Verify output file created
        assert test_config.data.output_path.exists()
        
        # Load and verify output
        output_df = pd.read_parquet(test_config.data.output_path)
        assert len(output_df) == 5
        assert "response" in output_df.columns
    
    @pytest.mark.asyncio
    async def test_process_with_checkpointing(self, test_config, mock_vllm_responses):
        """Test batch processor with checkpointing."""
        test_config.processing.batch_size = 2
        test_config.processing.checkpoint_interval = 1
        
        processor = BatchProcessor(test_config)
        
        # Create mock generation manager
        gen_manager = AsyncMock(spec=GenerationManager)
        gen_manager.generate_batch = AsyncMock(
            side_effect=lambda prompts, **kwargs: [
                {"choices": [{"text": f"Response for: {p}"}]} 
                for p in prompts
            ]
        )
        gen_manager.extract_texts_from_responses = lambda responses: [
            r["choices"][0]["text"] for r in responses
        ]
        gen_manager.get_statistics.return_value = {"total_requests": 5}
        
        # Process
        stats = await processor.process(
            gen_manager,
            dry_run=False,
            progress_bar=False
        )
        
        assert stats["total_processed"] == 5
        
        # Check checkpoints were created
        checkpoint_files = list(test_config.processing.checkpoint_dir.glob("checkpoint_*.json"))
        assert len(checkpoint_files) > 0


class TestGenerationPipeline:
    """Test GenerationPipeline functionality."""
    
    @pytest.fixture
    def test_config(self, temp_dir, sample_parquet_file):
        """Create test configuration."""
        return Config(
            data=DataConfig(
                input_path=sample_parquet_file,
                output_path=temp_dir / "output.parquet"
            ),
            models=[ModelConfig(url="http://localhost:8000")]
        )
    
    @pytest.mark.asyncio
    async def test_pipeline_validation(self, test_config):
        """Test pipeline validation."""
        pipeline = GenerationPipeline(test_config)
        
        # Mock the generation manager initialization
        with patch.object(pipeline, 'initialize', new_callable=AsyncMock) as mock_init:
            result = await pipeline.validate()
            
            assert result is True
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_validation_missing_input(self, test_config):
        """Test pipeline validation with missing input file."""
        test_config.data.input_path = Path("nonexistent.parquet")
        
        pipeline = GenerationPipeline(test_config)
        result = await pipeline.validate()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_pipeline_run_dry(self, test_config):
        """Test running pipeline in dry run mode."""
        pipeline = GenerationPipeline(test_config)
        
        # Mock health check
        with patch.object(pipeline, 'initialize', new_callable=AsyncMock):
            with patch.object(pipeline.batch_processor, 'process', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = {
                    "total_processed": 5,
                    "processing_time": 1.0,
                    "prompts_per_second": 5.0,
                    "model_statistics": {}
                }
                
                results = await pipeline.run(dry_run=True, progress_bar=False)
                
                assert results["total_processed"] == 5
                mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_models(self, test_config):
        """Test listing models from endpoints."""
        pipeline = GenerationPipeline(test_config)
        
        # Mock the clients
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=["model1", "model2"])
        
        with patch.object(pipeline, 'generation_manager') as mock_gen_manager:
            mock_gen_manager.clients = {"http://localhost:8000/": mock_client}
            mock_gen_manager.model_manager.endpoints = [
                MagicMock(url="http://localhost:8000/", name="test_model")
            ]
            
            models = await pipeline.list_models()
            
            assert "test_model" in models
            assert models["test_model"] == ["model1", "model2"]