import numpy as np

def generate(n_samples = 1000, noise=0.0):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    y = 21 * np.random.rand(n_samples)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.column_stack((x, y, z))
    if noise > 0.0:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, t