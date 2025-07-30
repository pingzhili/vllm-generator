#!/usr/bin/env python3
"""
Example of how to load and use the saved hidden states.
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys


def load_hidden_states(file_path):
    """Load hidden states from saved file."""
    results = torch.load(file_path, map_location='cpu')
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_hidden_states.py hidden_states.pt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Load results
    results = load_hidden_states(file_path)
    
    hidden_states = results['hidden_states']  # numpy array [n_samples, hidden_dim]
    texts = results['texts']  # list of strings
    
    print(f"Loaded hidden states:")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Hidden dimension: {results['hidden_dim']}")
    
    # Show first few texts
    print(f"\nFirst 3 texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i}: {text[:100]}...")
    
    # Compute similarity between first two texts
    if len(hidden_states) >= 2:
        similarity = cosine_similarity([hidden_states[0]], [hidden_states[1]])[0, 0]
        print(f"\nCosine similarity between first two texts: {similarity:.4f}")
    
    # Show some statistics
    print(f"\nHidden states statistics:")
    print(f"  Mean: {hidden_states.mean():.6f}")
    print(f"  Std: {hidden_states.std():.6f}")
    print(f"  Min: {hidden_states.min():.6f}")
    print(f"  Max: {hidden_states.max():.6f}")


if __name__ == "__main__":
    main()