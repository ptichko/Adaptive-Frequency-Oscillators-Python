"""Adaptive-Frequency Fitzhugh-Nagumo Oscillator

Righetti, Buchli, and Ijspeert (2006).
"""

import numpy as np


def fitzhugh_nagumo(t, x, a, b, e, F, F_t):
    """Fitzhugh-Nagumo oscillator with Hebbian frequency learning.

    Parameters
    ----------
    t : float
        Current time.
    x : array_like, shape (3,)
        State vector [X, Y, W] where W is the learned frequency.
    a, b : float
        Oscillator parameters.
    e : float
        Learning rate (e > 0).
    F : ndarray
        Input signal values.
    F_t : ndarray
        Time points corresponding to F.

    Returns
    -------
    ndarray, shape (3,)
        Derivatives [dX/dt, dY/dt, dW/dt].
    """
    F_val = np.interp(t, F_t, F)

    X, Y, W = x[0], x[1], x[2]

    r = np.sqrt(X**2 + Y**2) + np.finfo(float).eps

    dXdt = X * (X - a) * (1 - X) - Y + e * F_val
    dYdt = W * (X - b * Y)
    dWdt = -e * F_val * (Y / r)

    return np.array([dXdt, dYdt, dWdt])
