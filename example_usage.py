#!/usr/bin/env python3
"""
Example usage of vLLM Generator
"""

import pandas as pd
from pathlib import Path

# Create sample data
def create_sample_data():
    """Create a sample parquet file with questions"""
    
    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "How do neural networks work?",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "What is quantum computing?",
        "How does the internet work?",
        "What are the benefits of renewable energy?",
        "Explain machine learning to a 5-year-old.",
        "What is the difference between AI and ML?"
    ]
    
    # Create dataframe
    df = pd.DataFrame({
        "question": questions,
        "category": ["Geography", "Physics", "AI", "Environment", "Biology", 
                     "Technology", "Technology", "Environment", "AI", "AI"],
        "difficulty": ["Easy", "Hard", "Medium", "Medium", "Medium",
                      "Hard", "Medium", "Easy", "Easy", "Medium"]
    })
    
    # Save to parquet
    output_path = Path("sample_questions.parquet")
    df.to_parquet(output_path)
    print(f"Created sample data: {output_path}")
    return output_path


def example_basic_usage():
    """Example of basic usage"""
    print("\n=== Basic Usage Example ===")
    print("""
python -m vllm_generator \\
    --input sample_questions.parquet \\
    --model meta-llama/Llama-2-7b-hf \\
    --output results.parquet \\
    --max-tokens 100
    """)


def example_repeat_generation():
    """Example with repeat generation"""
    print("\n=== Repeat Generation Example ===")
    print("""
python -m vllm_generator \\
    --input sample_questions.parquet \\
    --model mistralai/Mistral-7B-v0.1 \\
    --num-repeats 5 \\
    --repeat-strategy temperature_schedule \\
    --temperature-schedule 0.3,0.5,0.7,0.9,1.1 \\
    --output-format nested \\
    --output results_multiple.parquet
    """)


def example_parallel_processing():
    """Example with parallel processing"""
    print("\n=== Parallel Processing Example ===")
    print("""
# Multi-server mode (requires multiple GPUs)
python -m vllm_generator \\
    --input large_dataset.parquet \\
    --model meta-llama/Llama-2-70b-hf \\
    --parallel-mode multi_server \\
    --num-workers 4 \\
    --worker-gpus 0,1,2,3,4,5,6,7 \\
    --tensor-parallel-size 2 \\
    --batch-size 8 \\
    --output parallel_results.parquet
    """)


def example_with_config():
    """Example using configuration file"""
    print("\n=== Configuration File Example ===")
    
    # Create example config
    config_content = """
model_config:
  model: meta-llama/Llama-2-7b-hf
  temperature: 0.7
  max_tokens: 200
  top_p: 0.95

generation_config:
  batch_size: 16
  num_repeats: 3
  checkpoint_frequency: 50
  error_handling: skip

data_config:
  question_column: question
  output_format: wide
  prompt_template: |
    Answer the following question concisely:
    {question}
    
    Answer:

logging_config:
  level: INFO
  progress_bar: true
  save_metadata: true
"""
    
    config_path = Path("example_config.yaml")
    config_path.write_text(config_content)
    print(f"Created config file: {config_path}")
    
    print("""
# Run with config file
python -m vllm_generator \\
    --config-file example_config.yaml \\
    --input sample_questions.parquet \\
    --output configured_results.parquet
    """)


def example_custom_preprocessing():
    """Example with custom preprocessing"""
    print("\n=== Custom Preprocessing Example ===")
    
    # Create preprocessing function
    preprocess_content = '''
def preprocess(question, row):
    """Add category context to questions"""
    category = row.get("category", "General")
    difficulty = row.get("difficulty", "Medium")
    
    return f"""[Category: {category}] [Difficulty: {difficulty}]
Question: {question}
Please provide a detailed answer appropriate for the difficulty level."""
'''
    
    preprocess_path = Path("custom_preprocess.py")
    preprocess_path.write_text(preprocess_content)
    print(f"Created preprocessing file: {preprocess_path}")
    
    print("""
# Run with custom preprocessing
python -m vllm_generator \\
    --input sample_questions.parquet \\
    --model gpt2 \\
    --preprocessing-fn custom_preprocess.py \\
    --output preprocessed_results.parquet
    """)


def example_with_chat_template():
    """Example using tokenizer's chat template"""
    print("\n=== Chat Template Example ===")
    print("""
# Using tokenizer's built-in chat template
python -m vllm_generator \\
    --input sample_questions.parquet \\
    --model meta-llama/Llama-2-7b-chat-hf \\
    --use-chat-template \\
    --system-prompt "You are a helpful assistant." \\
    --max-tokens 200 \\
    --output chat_results.parquet
    """)


def analyze_results():
    """Example of analyzing results"""
    print("\n=== Analyzing Results ===")
    print("""
import pandas as pd

# Load results
results = pd.read_parquet("results.parquet")

# For wide format with repeats
if "response_0" in results.columns:
    # Calculate response lengths
    for i in range(3):  # Assuming 3 repeats
        col = f"response_{i}"
        if col in results.columns:
            results[f"{col}_length"] = results[col].str.len()
    
    # Find average length per question
    length_cols = [col for col in results.columns if col.endswith("_length")]
    results["avg_response_length"] = results[length_cols].mean(axis=1)

# Display sample results
print(results[["question", "response_0"]].head())

# Save analysis
results.to_csv("results_analysis.csv", index=False)
    """)


if __name__ == "__main__":
    print("vLLM Generator - Example Usage")
    print("=" * 50)
    
    # Create sample data
    sample_file = create_sample_data()
    
    # Show examples
    example_basic_usage()
    example_repeat_generation()
    example_parallel_processing()
    example_with_config()
    example_with_chat_template()
    example_custom_preprocessing()
    analyze_results()
    
    print("\n" + "=" * 50)
    print("Note: These examples assume vLLM is installed and you have access to the specified models.")
    print("For testing without GPU, the system will automatically use a mock model.")
    print("\nFor more information, see the README.md file.")