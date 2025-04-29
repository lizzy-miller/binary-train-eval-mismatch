import numpy as np
import torch

def load_data(filepath, as_tensor = False):
    data = np.load(filepath)
    X = np.column_stack((data['x_c'], data['z_c']))
    y = data['y']
    
    if as_tensor:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y

def hamming_distance(predictions, targets, threshold=0.0, reduction="sum", as_tensor=True):
    """
    Compute Hamming Distance between predictions and targets.

    Args:
        predictions (Tensor): Model outputs (logits or probabilities).
        targets (Tensor): True binary labels (0 or 1).
        threshold (float): Threshold for binarization. Default is 0.0 (logits).
        reduction (str): 'mean' for fraction of mismatches, 'sum' for total mismatches.
        as_tensor (bool): If True, return a PyTorch scalar Tensor; if False, return a float.

    Returns:
        Scalar tensor or float: Hamming distance.
    """
    # Binarize predictions and targets
    preds_bin = (predictions > threshold).float()
    targets_bin = (targets > 0.5).float()

    # Compute elementwise mismatches
    mismatches = (preds_bin != targets_bin).float()

    # Apply reduction
    if reduction == "mean":
        result = mismatches.mean()
    elif reduction == "sum":
        result = mismatches.sum()
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}. Use 'mean' or 'sum'.")

    # Return as tensor or float
    if as_tensor:
        return result
    else:
        return result.item()
