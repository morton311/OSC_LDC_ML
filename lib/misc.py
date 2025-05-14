import numpy as np

def random_sample_series(series, n_samples, time_lag=64,train_ahead=5):
    """
    Randomly sample n_samples from a series.
    """
    if series.shape[0] - time_lag - train_ahead < n_samples:
        raise ValueError("n_samples must be less than the length of the series")
    
    indices = np.random.choice(series.shape[0] - time_lag - train_ahead, n_samples, replace=True)
    X = np.zeros((n_samples, time_lag, series.shape[-1]))
    Y = np.zeros((n_samples, train_ahead, series.shape[-1]))
    for i, idx in enumerate(indices):
        X[i] = series[idx:idx + time_lag]
        Y[i] = series[idx + time_lag:idx + time_lag + train_ahead]
    return X, Y