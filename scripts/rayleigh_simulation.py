"""Adaptive-Frequency Rayleigh Oscillator Simulation

Reproduces Figure 10 from:
Righetti, Buchli, and Ijspeert (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adafreq import rayleigh


def run():
    t0, tend = 0, 200
    Fs_ode = 20
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(20 * F_t)

    d, q, e = 50, 1, 0.3
    w0 = 20

    sol = solve_ivp(
        lambda t, x: rayleigh(t, x, d, q, e, F, F_t),
        [t0, tend], [0, 1, w0],         method='LSODA',
        t_eval=np.linspace(t0, tend, 5000),
        rtol=1e-6, atol=1e-8
    )
    t_out = sol.t
    y_out = sol.y
    F_out = np.interp(t_out, F_t, F)

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.plot(t_out, y_out[2], 'black')
    ax1.set_ylabel('W (Angular Frequency)')
    ax1.set_title('Adaptive-Frequency Rayleigh Oscillator')

    ax2_r = ax2.twinx()
    ax2_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax2_r.set_ylim([-4, 4])
    ax2_r.set_ylabel('F')
    ax2.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax2.set_ylim([-0.1, 0.1])
    ax2.set_xlim([0, 2])
    ax2.set_ylabel('X')

    ax3_r = ax3.twinx()
    ax3_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax3_r.set_ylim([-4, 4])
    ax3_r.set_ylabel('F')
    ax3.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax3.set_ylim([-0.1, 0.1])
    ax3.set_xlim([198, 200])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X')

    fig1.tight_layout()
    fig1.savefig('figures/rayleigh_simulation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Rayleigh simulation complete.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    run()
