"""Example usage of vLLM Generator."""

import asyncio
import pandas as pd
from pathlib import Path

from vllm_generator import Pipeline, Config
from vllm_generator.config import ConfigParser
from vllm_generator.pipeline import GenerationPipeline


async def basic_example():
    """Basic example with minimal configuration."""
    print("=== Basic Example ===")
    
    # Create sample data
    df = pd.DataFrame({
        "question": [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
        ],
        "id": [1, 2, 3]
    })
    df.to_parquet("sample_questions.parquet")
    
    # Create configuration
    config = ConfigParser.create_default_config(
        input_path="sample_questions.parquet",
        output_path="sample_responses.parquet",
        model_url="http://localhost:8000",
        num_samples=1,
        temperature=0.7,
        max_tokens=256
    )
    
    # Run pipeline
    pipeline = GenerationPipeline(config)
    results = await pipeline.run(dry_run=True)  # Use dry_run=True for testing
    
    print(f"Processed {results['total_processed']} items")
    print(f"Time: {results['processing_time']:.2f}s")
    print(f"Throughput: {results['prompts_per_second']:.2f} prompts/s")
    
    # Read results
    output_df = pd.read_parquet("sample_responses.parquet")
    print("\nResults:")
    print(output_df[["question", "response"]].head())


async def multi_sample_example():
    """Example with multiple samples per input."""
    print("\n=== Multi-Sample Example ===")
    
    # Configuration with multiple samples and temperature variation
    config_dict = {
        "data": {
            "input_path": "sample_questions.parquet",
            "output_path": "multi_sample_responses.parquet",
            "input_column": "question",
            "output_column": "response"
        },
        "models": [
            {"url": "http://localhost:8000", "name": "primary"}
        ],
        "generation": {
            "num_samples": 3,
            "temperature": [0.5, 0.8, 1.1],  # Different temperature for each sample
            "max_tokens": 512,
            "top_p": 0.95
        },
        "processing": {
            "batch_size": 16,
            "num_workers": 1
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    # Run pipeline
    pipeline = GenerationPipeline(config)
    results = await pipeline.run(dry_run=True)
    
    print(f"Generated {results['total_processed']} responses")
    
    # Read results - will have multiple rows per question
    output_df = pd.read_parquet("multi_sample_responses.parquet")
    print(f"\nTotal rows: {len(output_df)} (3 questions × 3 samples)")
    
    # Show samples for first question
    first_question_samples = output_df[output_df["id"] == 1]
    print("\nSamples for first question:")
    for idx, row in first_question_samples.iterrows():
        print(f"Sample {row['sample_idx']}: {row['response'][:50]}...")


async def multi_server_example():
    """Example with multiple vLLM servers."""
    print("\n=== Multi-Server Example ===")
    
    config_dict = {
        "data": {
            "input_path": "sample_questions.parquet",
            "output_path": "multi_server_responses.parquet"
        },
        "models": [
            {"url": "http://server1:8000", "name": "server1"},
            {"url": "http://server2:8000", "name": "server2"},
            {"url": "http://server3:8000", "name": "server3"}
        ],
        "generation": {
            "num_samples": 1,
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "processing": {
            "batch_size": 64,
            "num_workers": 3  # Parallel processing across servers
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    # Save configuration for reuse
    ConfigParser.to_yaml(config, Path("multi_server_config.yaml"))
    print("Saved configuration to multi_server_config.yaml")
    
    # In real usage, servers would be running
    # pipeline = GenerationPipeline(config)
    # results = await pipeline.run()


async def filtered_processing_example():
    """Example with data filtering and custom columns."""
    print("\n=== Filtered Processing Example ===")
    
    # Create sample data with categories
    df = pd.DataFrame({
        "question": [
            "What is Python?",
            "Explain quantum computing",
            "What is JavaScript?",
            "How do black holes form?",
            "What is Rust?",
        ],
        "id": [1, 2, 3, 4, 5],
        "category": ["programming", "physics", "programming", "physics", "programming"],
        "difficulty": [1, 3, 1, 3, 2]
    })
    df.to_parquet("categorized_questions.parquet")
    
    config_dict = {
        "data": {
            "input_path": "categorized_questions.parquet",
            "output_path": "filtered_responses.parquet",
            "input_column": "question",
            "output_column": "response",
            "copy_columns": ["id", "category", "difficulty"],
            "filter_condition": "category == 'programming' and difficulty <= 2",
            "shuffle": True
        },
        "models": [
            {"url": "http://localhost:8000"}
        ],
        "generation": {
            "temperature": 0.6,
            "max_tokens": 256
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    pipeline = GenerationPipeline(config)
    
    # This would process only programming questions with difficulty <= 2
    print("Will process filtered data: programming questions with difficulty <= 2")


async def checkpoint_resume_example():
    """Example showing checkpoint and resume functionality."""
    print("\n=== Checkpoint/Resume Example ===")
    
    config_dict = {
        "data": {
            "input_path": "sample_questions.parquet",
            "output_path": "checkpoint_example.parquet"
        },
        "models": [
            {"url": "http://localhost:8000"}
        ],
        "processing": {
            "batch_size": 1,  # Small batch for demonstration
            "checkpoint_interval": 1,  # Checkpoint after every batch
            "checkpoint_dir": "./example_checkpoints",
            "resume": False  # Set to True to resume from checkpoint
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    # First run - will create checkpoints
    print("First run - creating checkpoints...")
    # pipeline = GenerationPipeline(config)
    # await pipeline.run()
    
    # To resume from checkpoint:
    # config.processing.resume = True
    # pipeline = GenerationPipeline(config)
    # await pipeline.run()
    
    print("To resume: set config.processing.resume = True")


def create_sample_data():
    """Create various sample datasets for examples."""
    # Large dataset example
    large_df = pd.DataFrame({
        "question": [f"Question {i}: Tell me about topic {i}" for i in range(1000)],
        "id": list(range(1000)),
        "category": ["science", "tech", "history", "art"] * 250,
        "metadata": [{"source": "dataset", "version": 1}] * 1000
    })
    large_df.to_parquet("large_dataset.parquet")
    print("Created large_dataset.parquet with 1000 rows")


async def main():
    """Run all examples."""
    # Create sample data
    create_sample_data()
    
    # Run examples
    await basic_example()
    await multi_sample_example()
    await multi_server_example()
    await filtered_processing_example()
    await checkpoint_resume_example()
    
    print("\n=== Examples Complete ===")
    print("Note: These examples use dry_run=True for testing without a vLLM server")
    print("To run with actual vLLM server, set dry_run=False and ensure server is running")


if __name__ == "__main__":
    asyncio.run(main())