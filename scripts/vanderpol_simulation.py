"""Adaptive-Frequency Van der Pol Oscillator Simulations

Reproduces Figures 8 and 9 from:
Righetti, Buchli, and Ijspeert (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adafreq import vanderpol

Fs_ode = 20


def run():
    # %% Figure 8
    t0, tend = 0, 1000
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(30 * F_t)

    a, e = 50, 0.7
    w0_vals = [24, 28]

    fig8, ax8 = plt.subplots()
    for w0 in w0_vals:
        sol = solve_ivp(
            lambda t, x: vanderpol(t, x, a, e, F, F_t),
            [t0, tend], [0, 1, w0], method='LSODA',
            t_eval=np.linspace(t0, tend, 5000),
            rtol=1e-6, atol=1e-8
        )
        ax8.plot(sol.t, sol.y[2], 'black')
    ax8.set_title('Adaptive-Frequency Van der Pol Oscillator (Figure 8)')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('W (Angular Frequency)')
    ax8.legend([str(v) for v in w0_vals])
    fig8.savefig('figures/vanderpol_fig8.png', dpi=150, bbox_inches='tight')

    # %% Figure 9
    t0, tend = 0, 800
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(40 * F_t)

    a, e = 100, 0.7
    w0 = 40

    sol = solve_ivp(
        lambda t, x: vanderpol(t, x, a, e, F, F_t),
        [t0, tend], [1, 1, w0],         method='LSODA',
        t_eval=np.linspace(t0, tend, 5000),
        rtol=1e-6, atol=1e-8
    )
    t_out = sol.t
    y_out = sol.y
    F_out = np.interp(t_out, F_t, F)

    fig9, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.plot(t_out, y_out[2], 'black')
    ax1.set_ylim([35, 55])
    ax1.set_ylabel('W (Angular Frequency)')
    ax1.set_title('Adaptive-Frequency Van der Pol Oscillator (Figure 9)')

    ax2_r = ax2.twinx()
    ax2_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax2_r.set_ylim([-2.2, 2.2])
    ax2_r.set_ylabel('F')
    ax2.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax2.set_ylim([-2.2, 2.2])
    ax2.set_xlim([0, 1])
    ax2.set_ylabel('X')

    ax3_r = ax3.twinx()
    ax3_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax3_r.set_ylim([-2.2, 2.2])
    ax3_r.set_ylabel('F')
    ax3.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax3.set_ylim([-2.2, 2.2])
    ax3.set_xlim([799, 800])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X')

    fig9.tight_layout()
    fig9.savefig('figures/vanderpol_fig9.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Van der Pol simulations complete.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    run()
