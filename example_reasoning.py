"""Example of using vLLM Generator with reasoning models and thinking mode."""

import asyncio
import pandas as pd
from pathlib import Path

from vllm_generator.config import ConfigParser
from vllm_generator.pipeline import GenerationPipeline


async def reasoning_example():
    """Example using reasoning models with thinking mode enabled."""
    print("=== Reasoning Example with Thinking Mode ===")
    
    # Create sample reasoning problems
    df = pd.DataFrame({
        "problem": [
            "Solve for x: 2x + 5 = 13",
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "A train travels 120 miles in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
            "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        ],
        "id": [1, 2, 3, 4],
        "type": ["algebra", "logic", "word_problem", "pattern"]
    })
    df.to_parquet("reasoning_problems.parquet")
    
    # Configuration for reasoning with thinking
    config_dict = {
        "data": {
            "input_path": "reasoning_problems.parquet",
            "output_path": "reasoning_solutions.parquet",
            "input_column": "problem",
            "output_column": "solution",
            "copy_columns": ["id", "type"]
        },
        "models": [
            {"url": "http://localhost:8000", "name": "reasoning_model"}
        ],
        "generation": {
            "num_samples": 1,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 8192,  # Reasoning needs more tokens
            "presence_penalty": 1.5,
            "enable_thinking": True,  # Enable thinking mode
            "extra_body": {
                # Additional parameters if needed
                "repetition_penalty": 1.1
            }
        },
        "processing": {
            "batch_size": 4,
            "num_workers": 1
        },
        "retry": {
            "timeout": 600  # Longer timeout for reasoning
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    # Save configuration for reuse
    ConfigParser.to_yaml(config, Path("reasoning_config.yaml"))
    print("Saved configuration to reasoning_config.yaml")
    
    # Run pipeline (use dry_run=False with actual vLLM server)
    pipeline = GenerationPipeline(config)
    print("\nNote: This example uses dry_run mode. To run with actual reasoning:")
    print("1. Start vLLM with reasoning model: vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1")
    print("2. Run: python -m vllm_generator generate --config reasoning_config.yaml")
    
    # Show what the request would look like
    print("\nRequest format with enable_thinking:")
    print("""
{
    "prompt": "Solve for x: 2x + 5 = 13",
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": true
        },
        "repetition_penalty": 1.1
    }
}
    """)


async def multi_model_reasoning_example():
    """Example using multiple reasoning models with different thinking settings."""
    print("\n=== Multi-Model Reasoning Example ===")
    
    # Configuration with multiple models, some with thinking enabled
    config_dict = {
        "data": {
            "input_path": "reasoning_problems.parquet",
            "output_path": "multi_model_solutions.parquet",
            "input_column": "problem",
            "output_column": "solution"
        },
        "models": [
            {"url": "http://localhost:8000", "name": "model_with_thinking"},
            {"url": "http://localhost:8001", "name": "model_without_thinking"},
        ],
        "generation": {
            "num_samples": 2,  # 2 samples per problem
            "temperature": [0.6, 0.8],  # Different temperature for each sample
            "max_tokens": 4096,
            "enable_thinking": True,  # All models will use thinking mode
        },
        "processing": {
            "batch_size": 8,
            "num_workers": 2  # Parallel processing
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    print("Configuration for multi-model reasoning:")
    print("- 2 vLLM servers")
    print("- Thinking mode enabled for all")
    print("- 2 samples per problem with different temperatures")
    print("- Parallel processing with 2 workers")


async def custom_extra_body_example():
    """Example showing custom extra_body parameters."""
    print("\n=== Custom Extra Body Parameters Example ===")
    
    config_dict = {
        "data": {
            "input_path": "problems.parquet",
            "output_path": "solutions.parquet"
        },
        "models": [
            {"url": "http://localhost:8000"}
        ],
        "generation": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "enable_thinking": False,  # Disable thinking
            "extra_body": {
                # Custom parameters for vLLM
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "custom_mode": "fast"
                },
                "guided_json": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                },
                "custom_param": "value"
            }
        }
    }
    
    config = ConfigParser.from_dict(config_dict)
    
    print("Custom extra_body will be sent as:")
    print("""
{
    "extra_body": {
        "chat_template_kwargs": {
            "enable_thinking": false,
            "custom_mode": "fast"
        },
        "guided_json": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            }
        },
        "custom_param": "value"
    }
}
    """)


async def main():
    """Run all reasoning examples."""
    await reasoning_example()
    await multi_model_reasoning_example()
    await custom_extra_body_example()
    
    print("\n=== CLI Usage Examples ===")
    print("\n# Enable thinking mode via CLI:")
    print("python -m vllm_generator generate \\")
    print("    --input reasoning.parquet \\")
    print("    --output solutions.parquet \\")
    print("    --model-url http://localhost:8000 \\")
    print("    --enable-thinking \\")
    print("    --max-tokens 8192")
    
    print("\n# Use configuration file:")
    print("python -m vllm_generator generate --config reasoning_config.yaml")
    
    print("\n# Override thinking mode in config:")
    print("python -m vllm_generator generate \\")
    print("    --config base_config.yaml \\")
    print("    --enable-thinking")


if __name__ == "__main__":
    asyncio.run(main())