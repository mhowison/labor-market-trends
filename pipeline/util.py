import numpy as np

def weighted_mean(x, w):
    """Calculates the weighted mean."""
    return np.sum(x * w) / np.sum(w)

def weighted_covariance(x, y, w):
    """Calculates the weighted covariance."""
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_correlation(x, y, w):
    """Calculates the weighted Pearson correlation coefficient."""
    return weighted_covariance(x, y, w) / np.sqrt(weighted_covariance(x, x, w) * weighted_covariance(y, y, w))
