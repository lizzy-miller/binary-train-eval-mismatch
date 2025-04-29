import numpy as np
import os
import argparse
from scipy.special import expit

def generate_synthetic_data(n=1000000, noise_std=0.01, error_std=0.5, seed=None, filepath='data/synthetic_data.npz'):
    """
    Generate synthetic binary classification data with specified parameters.
    """

    if seed is not None:
        np.random.seed(seed)

    # Coefficients
    beta_0 = -3
    beta_1 = 0.5
    beta_2 = 1
    beta_3 = 0.1
    beta_4 = -0.3  # unobserved

    # Define means and covariance
    mu = [1, 0]
    std_x = 2
    std_z = 1
    corr_xz = 0.6
    sigma = [
        [std_x**2, corr_xz * std_x * std_z],
        [corr_xz * std_x * std_z, std_z**2]
    ]

    # Generate x and z
    x, z = np.random.multivariate_normal(mean=mu, cov=sigma, size=n).T

    w = np.random.normal(loc=0.1, scale=1, size=n)  # unobserved variable

    # Center x, z, w
    x_c = (x - np.mean(x)) / np.std(x)
    z_c = (z - np.mean(z)) / np.std(z)
    w_c = (w - np.mean(w)) / np.std(w)

    # Compute xb
    error = np.random.normal(loc=0, scale=error_std, size=n)
    xb = (beta_0
          + beta_1 * x_c
          + beta_2 * z_c
          + beta_3 * x_c**2
          + beta_4 * w_c
          + error)

    # Compute probabilities and add noise
    p = expit(xb)
    noise = np.random.normal(loc=0, scale=noise_std, size=n)
    p_noisy_clipped = np.clip(p + noise, 0, 1).ravel()

    # Sample binary labels
    y = np.random.binomial(n=1, p=p_noisy_clipped, size=n)

    # Generate ids
    ids = np.arange(1, n + 1)

    # Save
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(filepath, id=ids, x_c=x_c, z_c=z_c, y=y)
    print(f"Saved synthetic dataset to {filepath}")

def main():
    """
    Main function to parse command-line arguments and generate synthetic binary classification data.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic binary classification data.")
    parser.add_argument("--n", type=int, default=1000000, help="Number of samples to generate.")
    parser.add_argument("--filepath", type=str, required=True, help="Where to save the generated data (.npz file).")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Standard deviation of probability noise.")
    parser.add_argument("--error_std", type=float, default=0.5, help="Standard deviation of error term.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    generate_synthetic_data(
        n=args.n,
        noise_std=args.noise_std,
        error_std=args.error_std,
        seed=args.seed,
        filepath=args.filepath
    )

if __name__ == "__main__":
    main()
