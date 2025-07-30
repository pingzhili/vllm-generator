"""Pytest configuration and fixtures."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
import shutil
from typing import Dict, Any, List
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_parquet_file(temp_dir):
    """Create a sample parquet file for testing."""
    data = {
        "question": [
            "What is the capital of France?",
            "Explain quantum computing",
            "How do neural networks work?",
            "What is machine learning?",
            "Describe the water cycle"
        ],
        "id": [1, 2, 3, 4, 5],
        "category": ["geography", "physics", "ai", "ai", "science"],
        "difficulty": [1, 3, 3, 2, 2]
    }
    df = pd.DataFrame(data)
    
    file_path = temp_dir / "test_data.parquet"
    df.to_parquet(file_path)
    
    return file_path


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "data": {
            "input_path": "test_input.parquet",
            "output_path": "test_output.parquet",
            "input_column": "question",
            "output_column": "response"
        },
        "models": [
            {
                "url": "http://localhost:8000",
                "name": "test_model"
            }
        ],
        "generation": {
            "num_samples": 1,
            "temperature": 1.0,
            "max_tokens": 512
        },
        "processing": {
            "batch_size": 2,
            "num_workers": 1
        },
        "retry": {
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 300.0
        },
        "logging": {
            "level": "INFO"
        }
    }


@pytest.fixture
def mock_vllm_response():
    """Create a mock vLLM response."""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "text": "This is a test response",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_vllm_responses():
    """Create multiple mock vLLM responses."""
    responses = []
    for i in range(5):
        responses.append({
            "id": f"cmpl-{i}",
            "object": "text_completion",
            "created": 1234567890 + i,
            "model": "test-model",
            "choices": [
                {
                    "text": f"Response {i}: This is a test response",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        })
    return responses


@pytest.fixture
def mock_multi_sample_response():
    """Create a mock response with multiple samples."""
    return {
        "id": "cmpl-multi",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "text": f"Sample {i}: This is a test response",
                "index": i,
                "logprobs": None,
                "finish_reason": "stop"
            }
            for i in range(3)
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 24,
            "total_tokens": 34
        }
    }