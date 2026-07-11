"""Adaptive-Frequency Van der Pol Oscillator

Righetti, Buchli, and Ijspeert (2006).
"""

import numpy as np


def vanderpol(t, x, a, e, F, F_t):
    """Van der Pol oscillator with Hebbian frequency learning.

    Parameters
    ----------
    t : float
        Current time.
    x : array_like, shape (3,)
        State vector [X, Y, W] where W is the learned frequency.
    a : float
        Nonlinearity parameter (a > 0).
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

    dXdt = Y + e * F_val
    dYdt = -a * (X**2 - 1) * Y - W**2 * X
    dWdt = e * F_val * (Y / r)

    return np.array([dXdt, dYdt, dWdt])
