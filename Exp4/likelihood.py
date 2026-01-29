import numpy as np

def log_likelihood(p, data):
    return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))
