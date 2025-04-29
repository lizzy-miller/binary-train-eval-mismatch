import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from src.models.neural_net import NeuralNetwork_Binary
from src.utils import load_data

def flatten_params(model):
    """Flatten all model parameters into a single 1D tensor"""
    return torch.cat([param.view(-1) for param in model.parameters()])

def set_model_params(model, params):
    """Set model parameters from a 1D tensor"""
    idx = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(params[idx:idx + numel].view(param.size()))
        idx += numel

def main():
    parser = argparse.ArgumentParser(description="Loss landscape around a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (.pth)")
    parser.add_argument("--hidden_dims", type=int, nargs='+', required=True, help="Hidden dimensions of model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data for loss evaluation")
    parser.add_argument("--grid_size", type=int, default=21, help="Number of points per axis in the grid")
    parser.add_argument("--range", type=float, default=1.0, help="Range to perturb weights along each direction")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension of the model (number of features)")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="Normalize direction vectors (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Do not normalize direction vectors")
    parser.add_argument("--criterion", type=str, default="bce", help="Loss function to use: 'bce' (default), 'mse'")
    parser.add_argument("--dir_path", type=str, default=None, help="Path to precomputed random directions (.npz file)")
    parser.add_argument("--save_dirs", action='store_true', help="Save generated random directions")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory where results will be saved")
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    print("Running with configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Load model
    input_dim = args.input_dim
    hidden_dims = args.hidden_dims
    model = NeuralNetwork_Binary(input_dim, hidden_dims, output_dim=1)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Load data
    X, y = load_data(args.train_path, as_tensor= True)

    # Flatten model parameters
    w0 = flatten_params(model)

    # Create two random directions
    if args.dir_path is not None:
        print(f"Loading directions from {args.dir_path}")
        dirs = np.load(args.dir_path)
        d1 = torch.tensor(dirs['d1'], dtype=torch.float32)
        d2 = torch.tensor(dirs['d2'], dtype=torch.float32)
    else:
        d1 = torch.randn_like(w0)
        d2 = torch.randn_like(w0)

    # Normalize the directions if specified
    if args.normalize:
        d1 /= torch.norm(d1)
        d2 /= torch.norm(d2)

    # Save directions if needed
    if args.save_dirs and args.dir_path is None:
        os.makedirs(args.out_dir, exist_ok=True)  # Just in case
        base_name = os.path.splitext(os.path.basename(args.model_path))[0]
        dir_save_path = os.path.join(args.out_dir, f"{base_name}_directions.npz")
        np.savez(dir_save_path, d1=d1.numpy(), d2=d2.numpy())
        print(f"Random directions saved to {dir_save_path}")


    # Create the grid
    alphas = np.linspace(-args.range, args.range, args.grid_size)
    betas = np.linspace(-args.range, args.range, args.grid_size)
    losses = np.zeros((args.grid_size, args.grid_size))

    # Choose loss function
    if args.criterion.lower() == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {args.criterion}. Use 'bce' or 'mse'.")

    # Evaluate loss at each point
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbation = alpha * d1 + beta * d2
            new_params = w0 + perturbation
            set_model_params(model, new_params)

            with torch.no_grad():
                outputs = model(X)
                loss = criterion(outputs.squeeze(), y.squeeze())
                losses[i, j] = loss.item()

    # Save the loss landscape
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"{base_name}_loss_landscape.npz")

    np.savez(save_path, alphas=alphas, betas=betas, losses=losses)
    print(f"Loss landscape saved to {save_path}")
    print("Done.")

if __name__ == "__main__":
    main()

# This script computes the loss landscape around a trained binary classification model.
# It perturbs the model parameters along two random directions and evaluates the loss on a grid of points.
# The results are saved to a .npz file containing the alpha and beta values along with the corresponding losses.


    
        
    