"""NumPy mirror for checking the discrete regularizer without PyTorch."""

import numpy as np


def discrete_defects(y, u, h, d, dt=.125, f0=50.0):
    y = np.asarray(y)
    u = np.asarray(u)
    ybar = (y[..., 1:] + y[..., :-1]) / 2
    ubar = (u[..., 1:] + u[..., :-1]) / 2
    s = np.diff(y, axis=-1) / dt - (f0 * u[..., :-1] - d * ybar) / (2 * h)
    i = np.diff(y, axis=-1) - dt * (f0 * ubar - d * ybar) / (2 * h)
    return s, i
