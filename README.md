# vLLM Generator

A scalable and flexible text generation framework for processing dataframes using vLLM. Designed for research and production use cases requiring batch generation with support for multiple responses per input.

## Features

- 🚀 **High Performance**: Built on vLLM for efficient batch generation
- 🔄 **Repeat Generation**: Generate multiple responses per input with various strategies
- 🎯 **Flexible Output Formats**: Wide, long, or nested output formats
- ⚡ **Data Parallelism**: Multi-server and Ray-based distributed processing
- 💾 **Checkpoint & Resume**: Save progress and resume interrupted generations
- 📊 **Comprehensive Tracking**: Token usage, performance metrics, and progress monitoring
- 🔧 **Highly Configurable**: YAML/JSON configs, environment variables, and CLI args

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### With vLLM (requires GPU)
```bash
pip install vllm>=0.2.0
```

### With Ray (for distributed processing)
```bash
pip install ray>=2.0.0
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage
```bash
python -m vllm_generator \
  --input questions.parquet \
  --model meta-llama/Llama-2-7b-hf \
  --output results.parquet
```

### Repeat Generation
```bash
python -m vllm_generator \
  --input questions.parquet \
  --model mistralai/Mistral-7B-v0.1 \
  --num-repeats 5 \
  --temperature-schedule 0.5,0.7,0.9,1.1,1.3 \
  --repeat-strategy temperature_schedule
```

### Multi-GPU Parallel Processing
```bash
python -m vllm_generator \
  --input large_dataset.parquet \
  --model meta-llama/Llama-2-70b-hf \
  --parallel-mode multi_server \
  --num-workers 4 \
  --worker-gpus 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2
```

### Using Configuration File
```bash
python -m vllm_generator \
  --config-file configs/example_simple.yaml \
  --input data.parquet \
  --output results.parquet
```

## Input/Output Formats

### Input Format
The input should be a Parquet file with at least one column containing the questions/prompts:

```python
import pandas as pd

df = pd.DataFrame({
    "question": [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?"
    ],
    "category": ["AI", "Physics", "Biology"]  # Optional metadata
})
df.to_parquet("questions.parquet")
```

### Output Formats

#### Wide Format (default)
Each repeat gets its own column:
```
| question | response_0 | response_1 | response_2 |
|----------|------------|------------|------------|
| Q1       | R1_0       | R1_1       | R1_2       |
```

#### Long Format
Multiple rows per question:
```
| question | repeat_id | response |
|----------|-----------|----------|
| Q1       | 0         | R1_0     |
| Q1       | 1         | R1_1     |
```

#### Nested Format
List of responses in single column:
```
| question | response            |
|----------|---------------------|
| Q1       | [R1_0, R1_1, R1_2] |
```

## Configuration

### Configuration Priority
1. Configuration file (YAML/JSON)
2. Environment variables (prefixed with `VLLM_GEN_`)
3. Command line arguments

### Example Configuration File
```yaml
model_config:
  model: "meta-llama/Llama-2-7b-hf"
  temperature: 0.7
  max_tokens: 512

generation_config:
  batch_size: 32
  num_repeats: 3
  checkpoint_frequency: 100

parallel_config:
  mode: "single"
```

### Environment Variables
```bash
export VLLM_GEN_MODEL_CONFIG_TEMPERATURE=0.8
export VLLM_GEN_GENERATION_CONFIG_BATCH_SIZE=64
```

## Advanced Features

### Chat Templates
Use the model's built-in chat template for proper formatting:
```bash
python -m vllm_generator \
  --input questions.parquet \
  --model meta-llama/Llama-2-7b-chat-hf \
  --use-chat-template \
  --system-prompt "You are a helpful assistant." \
  --output chat_results.parquet
```

The `--use-chat-template` flag automatically applies the tokenizer's chat template, ensuring proper formatting for chat-tuned models.

### Custom Preprocessing
Create a preprocessing function:
```python
# preprocess.py
def preprocess(question, row):
    # Add context or modify question
    return f"Context: {row['context']}\n\nQuestion: {question}"
```

Use it:
```bash
python -m vllm_generator \
  --input data.parquet \
  --preprocessing-fn preprocess.py \
  --model gpt2
```

### Checkpointing
Automatically saves progress:
```bash
python -m vllm_generator \
  --input large_data.parquet \
  --checkpoint-frequency 100 \
  --resume-from-checkpoint outputs/checkpoint.json
```

### Token Tracking
Track token usage and costs:
```bash
python -m vllm_generator \
  --input data.parquet \
  --track-token-usage \
  --save-metadata
```

## Performance Tips

1. **Batch Size**: Start with 32 and adjust based on GPU memory
2. **Tensor Parallelism**: Use for models that don't fit on single GPU
3. **Data Parallelism**: Use multi_server mode for large datasets
4. **Sharding Strategy**: Use "balanced" for varying prompt lengths

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Lower `--gpu-memory-utilization`
- Enable `--cpu-offload-gb` for large models

### Slow Generation
- Increase `--batch-size` if GPU memory allows
- Use `--enable-prefix-caching` for similar prompts
- Enable `--parallel-mode multi_server` for multiple GPUs

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.