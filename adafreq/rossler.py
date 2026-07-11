"""Adaptive-Frequency Rossler Strange Attractor

Righetti, Buchli, and Ijspeert (2006).
"""

import numpy as np


def rossler(t, x, a, b, c, e, F, F_t):
    """Rossler attractor with Hebbian frequency learning.

    Parameters
    ----------
    t : float
        Current time.
    x : array_like, shape (4,)
        State vector [X, Y, Z, W] where W is the learned frequency.
    a, b, c : float
        Rossler oscillator parameters.
    e : float
        Learning rate (e > 0).
    F : ndarray
        Input signal values.
    F_t : ndarray
        Time points corresponding to F.

    Returns
    -------
    ndarray, shape (4,)
        Derivatives [dX/dt, dY/dt, dZ/dt, dW/dt].
    """
    F_val = np.interp(t, F_t, F)

    X, Y, Z, W = x[0], x[1], x[2], x[3]

    r = np.sqrt(X**2 + Y**2) + np.finfo(float).eps

    dXdt = -W * Y - Z + e * F_val
    dYdt = W * X + a * Y
    dZdt = b - c * Z + X * Z
    dWdt = -e * F_val * (Y / r)

    return np.array([dXdt, dYdt, dZdt, dWdt])
