#!/usr/bin/env python3
"""
Example script to use a trained MLP predictor.

Usage:
    python use_mlp_predictor.py --model-path ./mlp_model/best_model.pth --embedding [1.0,2.0,3.0,...]
"""

import argparse
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path


class MLPPredictor(nn.Module):
    """Two-layer MLP with GELU activation and sigmoid output."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


def load_model(model_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model architecture from checkpoint
    if 'args' in checkpoint:
        # Full checkpoint with args
        args = checkpoint['args']
        hidden_dim = args.hidden_dim
        model_state = checkpoint['model_state_dict']
    else:
        # Just state dict - infer dimensions
        model_state = checkpoint
        # Get input dim from first layer
        input_dim = model_state['layers.0.weight'].shape[1]
        hidden_dim = model_state['layers.0.weight'].shape[0]
    
    # Infer input dimension from model state
    input_dim = model_state['layers.0.weight'].shape[1]
    
    # Create model
    model = MLPPredictor(input_dim, hidden_dim)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, input_dim


def predict(model, embedding, device):
    """Make a prediction for a single embedding."""
    with torch.no_grad():
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        elif isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        embedding = embedding.unsqueeze(0).to(device)  # Add batch dimension
        prediction = model(embedding)
        return prediction.item()


def main():
    parser = argparse.ArgumentParser(description="Use trained MLP predictor")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--embedding", type=str, help="Embedding vector as JSON list")
    parser.add_argument("--embedding-file", type=str, help="Path to file containing embeddings")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, input_dim = load_model(args.model_path, device)
    print(f"Model loaded successfully (input_dim={input_dim})")
    
    if args.embedding:
        # Single embedding from command line
        try:
            embedding = json.loads(args.embedding)
            if len(embedding) != input_dim:
                raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {input_dim}")
            
            prediction = predict(model, embedding, device)
            print(f"\nPrediction: {prediction:.6f}")
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for embedding")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.embedding_file:
        # Multiple embeddings from file
        print(f"Loading embeddings from {args.embedding_file}...")
        with open(args.embedding_file, 'r') as f:
            embeddings = json.load(f)
        
        print(f"Making predictions for {len(embeddings)} embeddings...")
        predictions = []
        for i, embedding in enumerate(embeddings):
            if len(embedding) != input_dim:
                print(f"Warning: Skipping embedding {i} - dimension mismatch")
                continue
            
            pred = predict(model, embedding, device)
            predictions.append(pred)
            print(f"  Embedding {i}: {pred:.6f}")
        
        print(f"\nStatistics:")
        print(f"  Mean: {np.mean(predictions):.6f}")
        print(f"  Std:  {np.std(predictions):.6f}")
        print(f"  Min:  {np.min(predictions):.6f}")
        print(f"  Max:  {np.max(predictions):.6f}")
    
    else:
        print("Error: Provide either --embedding or --embedding-file")


if __name__ == "__main__":
    main()