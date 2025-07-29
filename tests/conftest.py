import pytest
import pandas as pd
import tempfile
from pathlib import Path
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    return pd.DataFrame({
        "question": [
            "What is the capital of France?",
            "Explain photosynthesis",
            "How do computers work?",
            "What is machine learning?",
            "Describe the water cycle"
        ],
        "category": ["Geography", "Biology", "Technology", "AI", "Science"],
        "difficulty": ["Easy", "Medium", "Hard", "Medium", "Easy"]
    })

@pytest.fixture
def sample_parquet_file(temp_dir, sample_dataframe):
    """Create a sample parquet file"""
    file_path = temp_dir / "test_questions.parquet"
    sample_dataframe.to_parquet(file_path)
    return file_path

@pytest.fixture
def mock_generation_results():
    """Mock generation results"""
    return [
        {
            "idx": 0,
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "tokens": 8,
            "latency": 0.1,
            "original_question": "What is the capital of France?"
        },
        {
            "idx": 1,
            "prompt": "Explain photosynthesis",
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "tokens": 15,
            "latency": 0.12,
            "original_question": "Explain photosynthesis"
        }
    ]

@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "model_config": {
            "model": "gpt2",
            "temperature": 0.7,
            "max_tokens": 100
        },
        "generation_config": {
            "batch_size": 2,
            "num_repeats": 1,
            "error_handling": "skip"
        },
        "data_config": {
            "question_column": "question",
            "output_format": "wide"
        }
    }