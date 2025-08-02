#!/usr/bin/env python3
"""
Extract hidden states from unique strings in a parquet file.

"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json


class MLPPredictor(nn.Module):
    """Two-layer MLP with GELU activation and sigmoid output."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Small dropout for regularization
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)

def load_unique_strings(parquet_path: Path, column: str):
    """Load unique strings from specified column in parquet file."""
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    unique_strings = df[column].dropna().unique().tolist()
    print(f"Found {len(unique_strings)} unique strings in column '{column}'")

    return unique_strings


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer, move model to CUDA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()

    return model, tokenizer


def apply_think_start_token(strings, tokenizer):
    """Apply chat template to strings."""
    print("Applying chat template...")

    appended_strings = []
    for text in tqdm(strings, desc="Templating"):
        appended_strings.append(text + "<think>\n\n")

    return appended_strings


def apply_adapter_on_last_hidden_states(model, tokenizer, texts, adapter):
    """Extract last hidden states from texts."""
    text_to_score = {}

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts)), desc="Processing texts"):
            text = texts[i]
            inputs = tokenizer(
                [text],
                return_tensors="pt",
            ).to(model.device)

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0, -1]  # [batch_size, seq_len, hidden_dim]
            adapter_score = adapter(last_hidden_states)
            text_to_score[text] = adapter_score.item()

    return text_to_score


def main():
    parser = argparse.ArgumentParser(
        description="Apply MLP predictor from unique strings in parquet file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("parquet_file", type=Path, help="Input parquet file")
    parser.add_argument("column", type=str, help="Column name to extract strings from")
    parser.add_argument("model_name", type=str, help="HuggingFace model name or path")
    parser.add_argument("output_file", type=Path, help="Output json file")
    parser.add_argument("adapter_path", type=Path, help="Output file for hidden states (.pt)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to process")

    args = parser.parse_args()

    # 1. Load unique strings from parquet
    unique_strings = load_unique_strings(args.parquet_file, args.column)

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 3. Apply chat template
    templated_strings = apply_think_start_token(unique_strings, tokenizer)

    # 4. Extract hidden states
    adapter = MLPPredictor(input_dim=model.config.hidden_size, hidden_dim=64).cuda()
    text_to_hidden_states = apply_adapter_on_last_hidden_states(model, tokenizer, templated_strings, adapter)

    # 5. Save results
    with open(args.output_file, "w") as f:
        json.dump(text_to_hidden_states, f, indent=4)

    print(f"\n✓ Processing complete!")
    print(f"✓ Input: {len(unique_strings)} unique strings from {args.parquet_file}")
    print(f"✓ Output: {args.output_file}")


if __name__ == "__main__":
    main()