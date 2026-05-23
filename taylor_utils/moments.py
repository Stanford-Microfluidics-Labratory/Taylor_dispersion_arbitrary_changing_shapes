import numpy as np


def weighted_mean(x):
    """Compute the mean along axis 0 (across particles), keeping dims."""
    return np.sum(x, axis=0, keepdims=True) / np.sum(np.ones_like(x), axis=0, keepdims=True)


def weighted_variance(x, weighted_x):
    """Compute the variance along axis 0 (across particles)."""
    return np.sum((x - weighted_x) ** 2, axis=0) / np.sum(np.ones_like(x), axis=0)


def weighted_skewness(x, weighted_x, weighted_var):
    """Compute the skewness along axis 0 (across particles)."""
    return np.sum((x - weighted_x) ** 3, axis=0) / np.sum(np.ones_like(x), axis=0) / weighted_var ** (3 / 2)


def weighted_kurtosis(x, weighted_x, weighted_var):
    """Compute the kurtosis along axis 0 (across particles)."""
    return np.sum((x - weighted_x) ** 4, axis=0) / np.sum(np.ones_like(x), axis=0) / weighted_var ** 2
