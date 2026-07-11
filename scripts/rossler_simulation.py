"""Adaptive-Frequency Rossler Strange Attractor Simulation

Reproduces Figure 10 from:
Righetti, Buchli, and Ijspeert (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adafreq import rossler

Fs_ode = 20


def run():
    t0, tend = 0, 500
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(20 * F_t)

    a, b, c, e = 0.15, 0.1, 8.5, 4
    w0 = 30

    sol = solve_ivp(
        lambda t, x: rossler(t, x, a, b, c, e, F, F_t),
        [t0, tend], [1, 0, 0, w0],         method='LSODA',
        t_eval=np.linspace(t0, tend, 10000),
        rtol=1e-6, atol=1e-8
    )
    t_out = sol.t
    y_out = sol.y
    F_out = np.interp(t_out, F_t, F)

    # %% Static Plots
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    ax1.plot(t_out, y_out[3], 'black')
    ax1.set_xlim([t0, tend])
    ax1.set_ylim([10, 32])
    ax1.set_ylabel('W (Angular Frequency)')
    ax1.set_title('Adaptive-Frequency Rossler Strange Attractor')

    ax2_r = ax2.twinx()
    ax2_r.plot(t_out, F_out, 'k--', linewidth=1.5, label='F')
    ax2_r.set_ylim([-2, 2])
    ax2_r.set_ylabel('F')
    ax2.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E', label='X')
    ax2.set_ylim([-2, 2])
    ax2.set_xlim([0, 2])
    ax2.set_ylabel('X')

    ax3_r = ax3.twinx()
    ax3_r.plot(t_out, F_out, 'k--', linewidth=1.5)
    ax3_r.set_ylim([-2, 2])
    ax3_r.set_ylabel('F')
    ax3.plot(t_out, y_out[0], linewidth=2, color='#7E2F8E')
    ax3.set_ylim([-200, 200])
    ax3.set_xlim([498, 500])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X')

    fig1.tight_layout()
    fig1.savefig('figures/rossler_static.png', dpi=150, bbox_inches='tight')

    # %% Animated GIF
    fig2 = plt.figure(figsize=(5, 8))
    ax4 = fig2.add_subplot(3, 1, 1, projection='3d')
    line4, = ax4.plot([], [], [], color='#808080', linewidth=0.25)
    dot4, = ax4.plot([], [], [], '.', markersize=5, color='#7E2F8E')
    ax4.set_xlim([-200, 200])
    ax4.set_ylim([-200, 200])
    ax4.set_zlim([0, 2500])
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_title('Phase Space')

    ax5 = fig2.add_subplot(3, 1, 2)
    line5, = ax5.plot([], [], color='#808080', linewidth=0.25)
    dot5, = ax5.plot([], [], '.', markersize=5, color='#7E2F8E')
    ax5.set_xlim([t0, tend])
    ax5.set_ylim([15, 32.5])
    ax5.axhline(20, color='k', linestyle='--')
    ax5.set_ylabel('w')
    ax5.set_title('Frequency Adaptation')

    ax6 = fig2.add_subplot(3, 1, 3)
    line6, = ax6.plot([], [], color='#808080', linewidth=0.25)
    dot6, = ax6.plot([], [], '.', markersize=5, color='#7E2F8E')
    ax6.set_xlim([t0, tend])
    ax6.set_ylim([-200, 200])
    ax6.set_ylabel('y')
    ax6.set_xlabel('Time')
    ax6.set_title('Oscillation (y component)')

    n_total = len(t_out)

    def update(frame):
        k = min(frame * 20, n_total - 1)
        line4.set_data(y_out[0, :k], y_out[1, :k])
        line4.set_3d_properties(y_out[2, :k])
        dot4.set_data([y_out[0, k]], [y_out[1, k]])
        dot4.set_3d_properties([y_out[2, k]])
        line5.set_data(t_out[:k], y_out[3, :k])
        dot5.set_data([t_out[k]], [y_out[3, k]])
        line6.set_data(t_out[:k], y_out[1, :k])
        dot6.set_data([t_out[k]], [y_out[1, k]])
        fig2.suptitle(f'Step {k}')
        return line4, dot4, line5, dot5, line6, dot6

    n_frames = int(n_total // 20)
    anim = FuncAnimation(fig2, update, frames=n_frames, blit=False)
    anim.save('figures/rossler_phasep.gif', writer=PillowWriter(fps=15))
    plt.close(fig2)

    plt.close('all')
    print("Rossler simulation complete.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    run()
