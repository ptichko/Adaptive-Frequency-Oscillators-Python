"""Adaptive-Frequency Fitzhugh-Nagumo Oscillator Simulation

Reproduces Figure 10 from:
Righetti, Buchli, and Ijspeert (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adafreq import fitzhugh_nagumo

Fs_ode = 20

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 11


def run():
    t0, tend = 0, 350
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(8 * F_t)

    a, b, e = -12, 0.01, 5
    w0 = 180

    sol = solve_ivp(
        lambda t, x: fitzhugh_nagumo(t, x, a, b, e, F, F_t),
        [t0, tend], [0, 1, w0], method='LSODA',
        t_eval=np.linspace(t0, tend, 5000),
        rtol=1e-6, atol=1e-8
    )
    t_out = sol.t
    y_out = sol.y
    F_out = np.interp(t_out, F_t, F)

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.plot(t_out, y_out[2], 'black')
    ax1.set_xlim([t0, tend])
    ax1.set_ylim([170, 240])
    ax1.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax1.set_title('Adaptive-Frequency Fitzhugh-Nagumo Oscillator', fontsize=FONT_TITLE)
    ax1.tick_params(labelsize=FONT_TICK)

    ax2_r = ax2.twinx()
    ax2_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax2_r.set_ylim([-5, 5])
    ax2_r.set_ylabel('F', fontsize=FONT_LABEL)
    ax2_r.tick_params(labelsize=FONT_TICK)
    ax2.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax2.set_ylim([-12, 12])
    ax2.set_xlim([0, 5])
    ax2.set_ylabel('X', fontsize=FONT_LABEL)
    ax2.tick_params(labelsize=FONT_TICK)

    ax3_r = ax3.twinx()
    ax3_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax3_r.set_ylim([-5, 5])
    ax3_r.set_ylabel('F', fontsize=FONT_LABEL)
    ax3_r.tick_params(labelsize=FONT_TICK)
    ax3.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax3.set_ylim([-12, 12])
    ax3.set_xlim([345, 350])
    ax3.set_xlabel('Time', fontsize=FONT_LABEL)
    ax3.set_ylabel('X', fontsize=FONT_LABEL)
    ax3.tick_params(labelsize=FONT_TICK)

    fig1.tight_layout()
    fig1.savefig('figures/fn_simulation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Fitzhugh-Nagumo simulation complete.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    run()
