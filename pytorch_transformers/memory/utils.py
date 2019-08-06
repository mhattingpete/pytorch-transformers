import math
import numpy as np

def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)
    