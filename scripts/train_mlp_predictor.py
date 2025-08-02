#!/usr/bin/env python3
"""
Train a simple 2-layer MLP predictor on embedding-value pairs.

Usage:
    python train_mlp_predictor.py --data-path data.pkl --output-dir ./mlp_model
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple
import wandb


class EmbeddingDataset(Dataset):
    """Dataset for embedding-value pairs."""
    
    def __init__(self, data_dict: Dict[torch.Tensor, float]):
        """Initialize dataset from dictionary."""
        self.embeddings = []
        self.values = []
        
        for embedding, value in data_dict.items():
            # Ensure embedding is a tensor
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            else:
                embedding = embedding.float()
            
            self.embeddings.append(embedding)
            self.values.append(float(value))
        
        # Stack all embeddings
        self.embeddings = torch.stack(self.embeddings)
        self.values = torch.tensor(self.values, dtype=torch.float32).unsqueeze(1)
        
        print(f"Dataset loaded: {len(self)} samples")
        print(f"Embedding shape: {self.embeddings[0].shape}")
        print(f"Value range: [{self.values.min():.4f}, {self.values.max():.4f}]")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.values[idx]


class MLPPredictor(nn.Module):
    """Two-layer MLP with GELU activation and sigmoid output."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (embeddings, targets) in enumerate(tqdm(dataloader)):
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(embeddings)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item()})
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for embeddings, targets in dataloader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate MAE as additional metric
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return avg_loss, mae


def main():
    parser = argparse.ArgumentParser(description="Train MLP predictor on embeddings")
    parser.add_argument("--data-path", type=str, required=True, help="Path to pickled data dictionary")
    parser.add_argument("--output-dir", type=str, default="./mlp_adapter", help="Output directory for model and logs")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    wandb.init(project="adaptive-thinking", name="mlp_predictor")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    data_dict = torch.load(args.data_path)
    
    # Create dataset
    dataset = EmbeddingDataset(data_dict)
    
    # Split dataset (90% train, 10% val)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Get input dimension
    sample_embedding, _ = dataset[0]
    input_dim = sample_embedding.shape[0]
    
    # Create model
    model = MLPPredictor(input_dim, args.hidden_dim).to(device)
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=args.num_epochs,
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pth'
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        wandb.log({"val_loss": val_loss, "val_mae": val_mae})


        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)

        curr_model_path = output_dir / f'model-epoch-{epoch}.pth'
        torch.save(model.state_dict(), curr_model_path)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'args': args
            }, best_model_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print()

    # Final evaluation on best model
    print(f"\nLoading best model from epoch {epoch - patience_counter + 1}...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_mae = evaluate(model, val_loader, criterion, device)
    print(f"Best model - Val Loss: {final_val_loss:.6f}, Val MAE: {final_val_mae:.6f}")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\nTraining complete! Models saved to {output_dir}")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training curves: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()