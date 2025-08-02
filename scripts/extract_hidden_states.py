#!/usr/bin/env python3
"""
Extract hidden states from unique strings in a parquet file.

Usage:
    python extract_hidden_states.py data.parquet question_column model_name output.pt
    python extract_hidden_states.py /root/open-math-reasoning/sample_c7_300.parquet problem Qwen/Qwen3-8B /root/open-math-reasoning/sample_c7_300_problem_last_hidden_states.pt
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


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


def apply_chat_template(strings, tokenizer):
    """Apply chat template to strings."""
    print("Applying chat template...")
    
    templated_strings = []
    for text in tqdm(strings, desc="Templating"):
        messages = [{"role": "user", "content": text}]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            templated = templated + "<think>\n\n"
        else:
            # Fallback if no chat template
            templated = text
            
        templated_strings.append(templated)
    
    return templated_strings


def extract_hidden_states(model, tokenizer, texts):
    """Extract last hidden states from texts."""
    text_to_hidden_states = {}
    
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
            # last_hidden_states = torch.stack([hs[0, -1] for hs in outputs.hidden_states]) # last token all layers
            print(f"Last hidden states shape: {last_hidden_states.shape}")
            text_to_hidden_states[text] = last_hidden_states.detach().cpu()
    
    return text_to_hidden_states


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from unique strings in parquet file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("parquet_file", type=Path, help="Input parquet file")
    parser.add_argument("column", type=str, help="Column name to extract strings from")
    parser.add_argument("model_name", type=str, help="HuggingFace model name or path")
    parser.add_argument("output_file", type=Path, help="Output file for hidden states (.pt)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to process")
    
    args = parser.parse_args()

    # 1. Load unique strings from parquet
    unique_strings = load_unique_strings(args.parquet_file, args.column)

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 3. Apply chat template
    templated_strings = apply_chat_template(unique_strings, tokenizer)

    # 4. Extract hidden states
    text_to_hidden_states = extract_hidden_states(model, tokenizer, templated_strings)

    # 5. Save results
    torch.save(text_to_hidden_states, args.output_file)

    print(f"\n✓ Processing complete!")
    print(f"✓ Input: {len(unique_strings)} unique strings from {args.parquet_file}")
    print(f"✓ Output: {args.output_file}")


if __name__ == "__main__":
    main()