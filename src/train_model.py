import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from src.models.neural_net import NeuralNetwork_Binary
import json
from datetime import datetime


def load_data(filepath):
    data = np.load(filepath)
    X = np.column_stack((data['x_c'], data['z_c']))
    y = data['y']
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    

def train_model(X_train, y_train, X_val, y_val, input_dim, hidden_dims=[10, 10], epochs=100, lr=1e-3):
    model = NeuralNetwork_Binary(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # === Training step ===
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # === Validation step ===
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a simple neural network on synthetic data.")

    parser.add_argument("--train_path", type=str, required=True, help="Path to training data (.npz)")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data (.npz)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[10, 10], help="Hidden layer dimensions (space-separated list)")
    parser.add_argument("--save_path", type=str, required=True, help="Full filepath to save the trained model (.pth)")
    args = parser.parse_args()
    
    if not os.path.exists(os.path.dirname(args.save_path)):
        raise ValueError(f"Directory '{os.path.dirname(args.save_path)}' does not exist. Please create it before saving.")


    X_train, y_train = load_data(args.train_path)
    X_val, y_val = load_data(args.val_path)
    print(f"Loaded training data from {args.train_path}")
    print(f"Loaded validation data from {args.val_path}")

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=X_train.shape[1],
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        lr=args.learning_rate
    )
    
    print("Successfully trained model")

    
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
    
    #Save Metadata 
    metadata = {
    "train_path": args.train_path,
    "val_path": args.val_path,
    "input_dim": X_train.shape[1],
    "hidden_dims": args.hidden_dims,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_file": args.save_path
    }
    
    metadata_save_path = args.save_path.replace(".pth", "_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_save_path}")

if __name__ == "__main__":
    main()

