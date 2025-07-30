# vLLM Generator

A scalable data generation pipeline for processing datasets through vLLM models with support for parallel processing via data splitting.

## Features

- **OpenAI API Integration**: Uses official OpenAI client for robust vLLM communication
- **Parquet I/O**: Load questions from parquet files and save responses
- **Individual Processing**: Process each question one-by-one (no batching complexity)
- **Repeated Sampling**: Generate multiple responses per input with temperature scheduling
- **Parallel Processing**: Split large datasets for parallel processing across multiple instances
- **Progress Tracking**: Real-time progress bars with loguru integration
- **Flexible Configuration**: YAML-based configuration with CLI overrides
- **Thinking Mode**: Support for reasoning models with `enable_thinking` parameter

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Start vLLM Server

```bash
vllm serve your-model --port 8000
```

### 2. Basic Usage

```bash
# Using configuration file
vllm-generator generate --config configs/example_simple.yaml

# Using CLI arguments
vllm-generator generate \
    --input data.parquet \
    --output results.parquet \
    --model-url http://localhost:8000 \
    --num-samples 3 \
    --temperature 0.8

# Using port shorthand
vllm-generator generate -c config.yaml --port 8001
```

### 3. Parallel Processing with Data Splits

For large datasets, split processing across multiple instances:

```bash
# Terminal 1: Process first quarter of data
vllm-generator generate -c config.yaml --split-id 1 --num-splits 4 --port 8000

# Terminal 2: Process second quarter
vllm-generator generate -c config.yaml --split-id 2 --num-splits 4 --port 8001

# Terminal 3: Process third quarter
vllm-generator generate -c config.yaml --split-id 3 --num-splits 4 --port 8002

# Terminal 4: Process fourth quarter
vllm-generator generate -c config.yaml --split-id 4 --num-splits 4 --port 8003
```

Each split will create its own output file (e.g., `output-1-of-4.parquet`, `output-2-of-4.parquet`, etc.).

## Configuration

### Basic Configuration

```yaml
data:
  input_path: "questions.parquet"
  output_path: "responses.parquet"
  input_column: "question"
  output_column: "response"

model:
  url: "http://localhost:8000"

generation:
  num_samples: 1
  temperature: 1.0
  max_tokens: 512
  enable_thinking: false  # For reasoning models

processing:
  batch_size: 32
```

### Advanced Features

#### Multiple Samples with Temperature Scheduling

```yaml
generation:
  num_samples: 5
  temperature: [0.5, 0.7, 0.9, 1.0, 1.2]  # Different temperature per sample
```

#### Enable Thinking Mode (for reasoning models)

```yaml
generation:
  enable_thinking: true
  max_tokens: 8192
```

#### Data Splitting for Parallel Processing

```yaml
processing:
  split_id: 1      # Which split to process (1-indexed)
  num_splits: 4    # Total number of splits
```

## CLI Arguments

```bash
vllm-generator generate --help

Options:
  --config, -c          Path to YAML configuration file
  --input, -i          Input parquet file
  --output, -o         Output parquet file
  --model-url, -m      vLLM server URL
  --port, -p           vLLM server port (shorthand for localhost)
  --num-samples, -n    Number of samples per input
  --temperature, -t    Sampling temperature
  --enable-thinking    Enable thinking mode for reasoning models
  --split-id           Split ID to process (1-indexed)
  --num-splits         Total number of splits
  --batch-size, -b     Batch size for processing
  --resume             Resume from checkpoint
  --dry-run            Run without API calls
```

## Parallel Processing Script

Use the provided script for easy parallel processing:

```bash
./scripts/parallel_split_processing.sh
```

This starts multiple instances processing different data splits in parallel.

## Examples

See the `configs/` directory for example configurations:
- `example_simple.yaml` - Basic single-sample generation
- `example_repeat_generation.yaml` - Multiple samples with temperature variation
- `example_reasoning.yaml` - Configuration for reasoning models with thinking mode
- `example_split_processing.yaml` - Parallel processing with data splits

## Architecture

The pipeline follows a clean, simple architecture:

1. **DataLoader**: Loads parquet files and handles data splitting
2. **VLLMClient**: Uses OpenAI API client for robust vLLM communication
3. **SimpleProcessor**: Processes each item individually (no batching)
4. **DataWriter**: Saves results with split-aware file naming

## OpenAI API Integration

The vLLM client now uses the official OpenAI Python client for reliable communication:

```python
from openai import OpenAI

# This is how vLLMClient connects internally
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require a real API key
    base_url="http://localhost:8000/v1"  # vLLM server endpoint
)

# Uses chat completions API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # vLLM uses whatever model is loaded
    messages=[{"role": "user", "content": "Your prompt"}],
    max_tokens=512,
    temperature=0.7,
    extra_body={
        "top_k": 20,  # vLLM-specific parameters
        "chat_template_kwargs": {"enable_thinking": True}
    }
)
```

Benefits:
- **Robust Error Handling**: Built-in retry logic and proper exception types
- **Connection Pooling**: Efficient HTTP connection management
- **Standard Interface**: Familiar OpenAI API patterns
- **Type Safety**: Full type hints and validation

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/
```