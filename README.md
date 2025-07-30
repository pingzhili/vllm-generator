# vLLM Generator

A scalable and efficient data generation pipeline for processing datasets using vLLM models. Supports batch processing, multiple model endpoints, repeated sampling, and fault-tolerant execution with checkpointing.

## Features

- 🚀 **High Performance**: Async/await based architecture with parallel processing
- 🔄 **Multiple Sampling**: Generate multiple responses per input with temperature scheduling
- ⚖️ **Load Balancing**: Distribute requests across multiple vLLM servers
- 💾 **Checkpointing**: Resume interrupted jobs from the last checkpoint
- 📊 **Progress Tracking**: Real-time progress bars and performance metrics
- 🛡️ **Fault Tolerant**: Automatic retries with exponential backoff
- 📝 **Comprehensive Logging**: Structured logging with loguru
- 🎯 **Flexible Configuration**: YAML configs with CLI override support

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```bash
# Generate responses for a parquet file
python -m vllm_generator generate \
    --input data/questions.parquet \
    --output data/responses.parquet \
    --model-url http://localhost:8000 \
    --temperature 0.7 \
    --max-tokens 512
```

### 2. Using Configuration File

```bash
# Generate using a YAML configuration
python -m vllm_generator generate --config configs/example_simple.yaml

# Override config parameters via CLI
python -m vllm_generator generate \
    --config configs/example_simple.yaml \
    --num-samples 3 \
    --temperature 0.9
```

### 3. Multiple Samples per Input

```yaml
# configs/multi_sample.yaml
generation:
  num_samples: 5
  temperature: [0.5, 0.7, 0.9, 1.1, 1.3]  # Different temperature for each sample
```

```bash
python -m vllm_generator generate --config configs/multi_sample.yaml
```

### 4. Multiple vLLM Servers

```bash
# Distribute load across multiple servers
python -m vllm_generator generate \
    --input large_dataset.parquet \
    --output results.parquet \
    --model-urls http://gpu1:8000 http://gpu2:8000 http://gpu3:8000 \
    --num-workers 6 \
    --batch-size 128
```

## Configuration

### YAML Configuration Structure

```yaml
data:
  input_path: "data/questions.parquet"
  output_path: "data/responses.parquet"
  input_column: "question"
  output_column: "response"
  copy_columns: ["id", "metadata"]  # Additional columns to preserve
  filter_condition: "category == 'science'"  # Optional pandas query
  limit: 1000  # Optional row limit
  shuffle: true  # Shuffle data before processing

models:
  - url: "http://localhost:8000"
    name: "primary_model"
    api_key: "optional-api-key"
    headers:
      Authorization: "Bearer token"

generation:
  num_samples: 3
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  max_tokens: 1024
  stop_sequences: ["###", "END"]
  seed: 42
  presence_penalty: 0.1
  frequency_penalty: 0.1
  enable_thinking: false  # Enable thinking mode for reasoning models
  extra_body:  # Additional vLLM parameters
    custom_param: "value"

processing:
  batch_size: 64
  num_workers: 2
  checkpoint_interval: 100
  checkpoint_dir: "./checkpoints"
  resume: false

retry:
  max_retries: 5
  retry_delay: 2.0
  timeout: 600
  backoff_factor: 2.0

logging:
  level: "INFO"
  file: "logs/generation.log"
  rotation: "100 MB"
  retention: "7 days"
```

## CLI Commands

### Generate Command

```bash
python -m vllm_generator generate [OPTIONS]
```

Key options:
- `--config`: Path to YAML configuration file
- `--input/--output`: Input/output parquet file paths
- `--model-url`: Single vLLM server URL
- `--model-urls`: Multiple vLLM server URLs
- `--num-samples`: Number of samples per input
- `--temperature`: Sampling temperature
- `--enable-thinking`: Enable thinking mode for reasoning models
- `--batch-size`: Batch size for processing
- `--num-workers`: Number of parallel workers
- `--resume`: Resume from checkpoint
- `--dry-run`: Test without making API calls

### Validate Command

```bash
# Validate configuration file
python -m vllm_generator validate --config config.yaml
```

### List Models Command

```bash
# List available models from vLLM servers
python -m vllm_generator list-models --model-url http://localhost:8000
```

## Advanced Usage

### Checkpointing and Resume

```bash
# First run - will save checkpoints
python -m vllm_generator generate --config config.yaml

# If interrupted, resume from last checkpoint
python -m vllm_generator generate --config config.yaml --resume
```

### Data Filtering

```yaml
data:
  filter_condition: "difficulty >= 3 and category == 'technical'"
  limit: 10000
  shuffle: true
```

### Temperature Scheduling

```yaml
generation:
  num_samples: 5
  # Each sample uses a different temperature
  temperature: [0.5, 0.7, 0.9, 1.1, 1.3]
```

### Custom Headers and Authentication

```yaml
models:
  - url: "http://api.example.com/v1"
    name: "api_endpoint"
    api_key: "${API_KEY}"  # Can use environment variables
    headers:
      X-Custom-Header: "value"
```

### Reasoning Mode with Thinking

For models that support reasoning with thinking (like DeepSeek-R1), you can enable thinking mode:

```bash
# Via CLI
python -m vllm_generator generate \
    --config config.yaml \
    --enable-thinking

# Via configuration
generation:
  enable_thinking: true
  max_tokens: 8192  # Reasoning may need more tokens
  top_k: 20
  presence_penalty: 1.5
```

The `enable_thinking` parameter is passed to vLLM via `extra_body` as:
```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": true
    }
  }
}
```

## Output Format

### Single Sample Output
```
| question | response | id | metadata |
|----------|----------|----|----------|
| What is ML? | Machine learning is... | 1 | {...} |
```

### Multiple Samples Output
```
| question | response | id | sample_idx |
|----------|----------|----|------------|
| What is ML? | Response 1... | 1 | 0 |
| What is ML? | Response 2... | 1 | 1 |
| What is ML? | Response 3... | 1 | 2 |
```

## Performance Tips

1. **Batch Size**: Larger batches (64-256) are more efficient for vLLM
2. **Workers**: Set workers to number of vLLM servers for optimal distribution
3. **Checkpointing**: Set checkpoint interval based on your dataset size
4. **Memory**: Monitor memory usage with large datasets or many samples

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=vllm_generator tests/
```

### Project Structure

```
vllm-generator/
├── src/vllm_generator/
│   ├── config/         # Configuration management
│   ├── data/          # Data loading and writing
│   ├── models/        # vLLM client and generation
│   ├── pipeline/      # Pipeline orchestration
│   ├── tracking/      # Metrics and monitoring
│   └── utils/         # Utilities and helpers
├── configs/           # Example configurations
├── tests/            # Unit tests
└── example_usage.py  # Usage examples
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure vLLM server is running and accessible
2. **Memory Error**: Reduce batch size or use chunked processing
3. **Timeout Error**: Increase timeout in retry configuration
4. **Invalid Column**: Check input_column exists in your parquet file

### Debug Mode

```bash
# Enable debug logging
python -m vllm_generator generate --config config.yaml --log-level DEBUG
```

## License

MIT License - see LICENSE file for details.