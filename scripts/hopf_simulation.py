"""Adaptive-Frequency Hopf Oscillator Simulations

Reproduces Figure 2 and extensions from:
Righetti, Buchli, and Ijspeert (2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adafreq import hopf

Fs_ode = 20

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 11


def run():
    # %% Figure 2: Multiple Learning Rates
    t0, tend = 0, 3000
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.cos(30 * F_t)

    m = 1
    e_vals = [1, 0.8, 0.6, 0.4]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for e in e_vals:
        sol = solve_ivp(
            lambda t, x: hopf(t, x, m, e, F, F_t),
            [t0, tend], [0, 1, 40], method='LSODA',
            t_eval=np.linspace(t0, tend, 5000),
            rtol=1e-6, atol=1e-8
        )
        ax1.plot(sol.t, sol.y[2], 'black')
    ax1.set_title('Adaptive-Frequency Hopf Oscillator with Hebbian Learning', fontsize=FONT_TITLE)
    ax1.set_xlabel('Time', fontsize=FONT_LABEL)
    ax1.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.legend([str(v) for v in e_vals], loc='best')
    fig1.tight_layout()
    fig1.savefig('figures/hopf_fig2_learning_rates.png', dpi=150, bbox_inches='tight')

    # %% Multiple Initial Conditions
    t0, tend = 0, 800
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.cos(30 * F_t)

    m, e = 1, 1
    w0_vals = [18, 26, 36, 42]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for w0 in w0_vals:
        sol = solve_ivp(
            lambda t, x: hopf(t, x, m, e, F, F_t),
            [t0, tend], [0, 1, w0], method='LSODA',
            t_eval=np.linspace(t0, tend, 5000),
            rtol=1e-6, atol=1e-8
        )
        ax2.plot(sol.t, sol.y[2], 'black')
    ax2.set_title('Adaptive-Frequency Hopf Oscillator with Hebbian Learning', fontsize=FONT_TITLE)
    ax2.set_xlabel('Time', fontsize=FONT_LABEL)
    ax2.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.legend([str(v) for v in w0_vals], loc='best')
    fig2.tight_layout()
    fig2.savefig('figures/hopf_fig2_initial_conditions.png', dpi=150, bbox_inches='tight')

    # %% One Rhythmic Frequency
    t0, tend = 0, 250
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.cos(3 * F_t)

    m, e = 1, 1
    sol = solve_ivp(
        lambda t, x: hopf(t, x, m, e, F, F_t),
        [t0, tend], [0, 1, 10], method='LSODA',
        t_eval=np.linspace(t0, tend, 10000),
        rtol=1e-6, atol=1e-8
    )
    t_out = sol.t
    y_out = sol.y
    F_out = np.interp(t_out, F_t, F)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(t_out, y_out[2], 'black')
    ax3.set_title('Adaptive-Frequency Hopf Oscillator with Hebbian Learning', fontsize=FONT_TITLE)
    ax3.set_xlabel('Time', fontsize=FONT_LABEL)
    ax3.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax3.tick_params(labelsize=FONT_TICK)
    fig3.tight_layout()
    fig3.savefig('figures/hopf_frequency_adaptation.png', dpi=150, bbox_inches='tight')

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax4a.plot(t_out, y_out[2], 'black', linewidth=1.5)
    ax4a.set_xlim([110, 160])
    ax4a.set_title('Dynamics of Frequency Adaptation', fontsize=FONT_TITLE)
    ax4a.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax4a.tick_params(labelsize=FONT_TICK)

    ax4b.plot(t_out, F_out, 'k--', linewidth=1.5, label='F')
    ax4b.plot(t_out, y_out[1], linewidth=2, color='#7E2F8E', label='Oscillator')
    ax4b.set_xlim([110, 160])
    ax4b.set_title('Input Signal and Oscillator', fontsize=FONT_TITLE)
    ax4b.set_xlabel('Time', fontsize=FONT_LABEL)
    ax4b.set_ylabel('Amplitude', fontsize=FONT_LABEL)
    ax4b.tick_params(labelsize=FONT_TICK)
    ax4b.legend(loc='best')
    fig4.tight_layout()
    fig4.savefig('figures/hopf_time_series.png', dpi=150, bbox_inches='tight')

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.plot(y_out[0], y_out[1], color='#808080', linewidth=0.5)
    ax5.plot(y_out[0, 0], y_out[1, 0], '.-', markersize=20, color='#7E2F8E')
    ax5.plot(y_out[0, -1], y_out[1, -1], '.-', markersize=20)
    ax5.set_title(f'Phase Portrait: e = {e}', fontsize=FONT_TITLE)
    ax5.set_xlabel('x', fontsize=FONT_LABEL)
    ax5.set_ylabel('y', fontsize=FONT_LABEL)
    ax5.tick_params(labelsize=FONT_TICK)
    fig5.tight_layout()
    fig5.savefig('figures/hopf_phase_portrait.png', dpi=150, bbox_inches='tight')

    # %% Animated GIF
    gif_start = np.searchsorted(t_out, 120)
    gif_stop = np.searchsorted(t_out, 145)

    fig6, (ax6a, ax6b, ax6c) = plt.subplots(3, 1, figsize=(5, 8))
    fig6.subplots_adjust(hspace=0.5)
    line6a, = ax6a.plot([], [], color='#808080', linewidth=0.25)
    dot6a, = ax6a.plot([], [], '.', markersize=20, color='#7E2F8E')
    ax6a.set_xlim(-1.5, 1.5)
    ax6a.set_ylim(-1.5, 1.5)
    ax6a.set_xlabel('x', fontsize=FONT_LABEL)
    ax6a.set_ylabel('y', fontsize=FONT_LABEL)
    ax6a.set_title('Phase Space', fontsize=FONT_TITLE)
    ax6a.tick_params(labelsize=FONT_TICK)

    line6b, = ax6b.plot([], [], color='#808080', linewidth=0.25)
    dot6b, = ax6b.plot([], [], '.', markersize=20, color='#7E2F8E')
    ax6b.set_xlim(t_out[gif_start], t_out[gif_stop])
    ax6b.set_ylim(2, 10)
    ax6b.axhline(3, color='k', linestyle='--')
    ax6b.set_ylabel('w', fontsize=FONT_LABEL)
    ax6b.set_title('Frequency Adaptation', fontsize=FONT_TITLE)
    ax6b.tick_params(labelsize=FONT_TICK)

    line6c, = ax6c.plot([], [], linewidth=1, color='#7E2F8E')
    line6d, = ax6c.plot([], [], '--', linewidth=1, color='#808080')
    ax6c.set_xlim(t_out[gif_start], t_out[gif_stop])
    ax6c.set_ylim(-1.2, 1.2)
    ax6c.set_ylabel('y', fontsize=FONT_LABEL)
    ax6c.set_xlabel('Time', fontsize=FONT_LABEL)
    ax6c.set_title('Oscillation (y-var) and Input Signal (dashed)', fontsize=FONT_TITLE)
    ax6c.tick_params(labelsize=FONT_TICK)

    def update(frame):
        k = gif_start + frame * 5
        if k >= gif_stop:
            k = gif_stop
        line6a.set_data(y_out[0, :k], y_out[1, :k])
        dot6a.set_data([y_out[0, k]], [y_out[1, k]])
        line6b.set_data(t_out[:k], y_out[2, :k])
        dot6b.set_data([t_out[k]], [y_out[2, k]])
        line6c.set_data(t_out[:k], y_out[1, :k])
        line6d.set_data(t_out[:k], F_out[:k])
        fig6.suptitle(f'Step {k}', fontsize=FONT_TITLE)
        return line6a, dot6a, line6b, dot6b, line6c, line6d

    n_frames = int((gif_stop - gif_start) // 5)
    anim = FuncAnimation(fig6, update, frames=n_frames, blit=False)
    anim.save('figures/hopf_phasep.gif', writer=PillowWriter(fps=15))
    plt.close(fig6)

    # %% Multiple Rhythmic Frequencies
    t0, tend = 0, 50
    f1, f2, f3 = 3, 6, 9
    F_t = np.linspace(t0, tend, int(tend * Fs_ode))
    F = np.sin(f1 * F_t) + np.cos(f2 * F_t) + np.sin(f3 * F_t)

    m, e = 1, 1
    w0_vals = [1, 4, 5, 10]
    y_comp_list = []

    fig7, ax7 = plt.subplots(figsize=(8, 5))
    for w0 in w0_vals:
        sol = solve_ivp(
            lambda t, x: hopf(t, x, m, e, F, F_t),
            [t0, tend], [0, 1, w0], method='LSODA',
            t_eval=np.linspace(t0, tend, 5000),
            rtol=1e-6, atol=1e-8
        )
        ax7.plot(sol.t, sol.y[2], 'black')
        y_comp_list.append((sol.t, sol.y[1]))
    ax7.axhline(f1, color='k', linestyle='--')
    ax7.axhline(f2, color='k', linestyle='--')
    ax7.axhline(f3, color='k', linestyle='--')
    ax7.set_title('Adaptive-Frequency Hopf Oscillator with Hebbian Learning', fontsize=FONT_TITLE)
    ax7.set_xlabel('Time', fontsize=FONT_LABEL)
    ax7.set_ylabel('W (Angular Frequency)', fontsize=FONT_LABEL)
    ax7.tick_params(labelsize=FONT_TICK)
    ax7.legend([str(v) for v in w0_vals], loc='best')
    fig7.tight_layout()
    fig7.savefig('figures/hopf_multi_freq.png', dpi=150, bbox_inches='tight')

    # Use last solution's time base for comparison plots
    t_multi = sol.t
    F_multi = np.interp(t_multi, F_t, F)
    fig8, axes8 = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for i, ax in enumerate(axes8):
        idx = len(w0_vals) - 1 - i
        t_i, y_i = y_comp_list[idx]
        ax.plot(t_i, np.interp(t_i, F_t, F), 'k-', linewidth=1.5, label='F')
        ax.plot(t_i, y_i, linewidth=2, color='#D95319', label='Oscillator')
        ax.set_ylabel('Amplitude', fontsize=FONT_LABEL)
        ax.set_title(f'{w0_vals[idx]}-Hz Oscillator', fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
        if i == 0:
            ax.legend(loc='best')
    axes8[-1].set_xlim([1, 40])
    axes8[-1].set_xlabel('Time', fontsize=FONT_LABEL)
    fig8.tight_layout()
    fig8.savefig('figures/hopf_multi_freq_comparison.png', dpi=150, bbox_inches='tight')

    plt.close('all')
    print("Hopf simulations complete.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    run()
