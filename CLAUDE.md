# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install from requirements
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/vllm_generator

# Run specific test types
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Running the Application
```bash
# Basic generation with config file
vllm-generator generate --config configs/qwen3-thinking.yaml

# Generate with CLI arguments
vllm-generator generate -i input.parquet -o output.parquet -m http://localhost:8000

# Parallel processing with data splits
vllm-generator generate -c config.yaml --split-id 1 --num-splits 4

# Validate configuration
vllm-generator validate --config config.yaml

# List available models
vllm-generator list-models --model-url http://localhost:8000
```

## Architecture Overview

### Core Components
- **Pipeline System**: `src/vllm_generator/pipeline/` - Orchestrates the entire generation process
  - `GenerationPipeline`: Main coordinator that initializes and runs the complete pipeline
  - `SimpleProcessor`: Handles data processing and response generation logic
- **Configuration Management**: `src/vllm_generator/config/` - Handles YAML config parsing and validation using Pydantic schemas
- **Data Layer**: `src/vllm_generator/data/` - Data loading (Parquet), processing, and writing operations
- **Model Interface**: `src/vllm_generator/models/vllm_client.py` - OpenAI-compatible client for vLLM servers
- **Utilities**: `src/vllm_generator/utils/` - Logging setup and helper functions

### Key Features
- **Data Splitting**: Supports parallel processing by splitting datasets across multiple processes/servers
- **Thinking Mode**: Special handling for reasoning models that generate internal thoughts before responses
- **Retry Logic**: Configurable retry mechanisms with exponential backoff for failed API calls
- **Metrics Tracking**: Built-in performance and error tracking via `src/vllm_generator/tracking/`

### Configuration System
Uses YAML configuration files with Pydantic validation. Key sections:
- `data`: Input/output paths and column mappings
- `model`: vLLM server URL and connection settings  
- `generation`: Sampling parameters (temperature, top_p, max_tokens, etc.)
- `processing`: Data splitting and parallel processing options
- `retry`: Failure handling and retry configuration
- `logging`: Structured logging with rotation and retention

### Entry Points
- CLI: `vllm-generator` command (via `src/vllm_generator/__main__.py`)
- Module: `python -m vllm_generator`
- Programmatic: Import `GenerationPipeline` from `src/vllm_generator.pipeline`