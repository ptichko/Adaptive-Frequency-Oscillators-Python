# Adaptive-Frequency Oscillators (Python)

Python implementation of adaptive-frequency oscillators based on dynamic Hebbian learning, reproducing results from:

> **Righetti, Buchli, & Ijspeert (2006).** *Dynamic Hebbian learning in adaptive frequency oscillators.* Physica D. [DOI: 10.1016/j.physd.2006.02.009](https://doi.org/10.1016/j.physd.2006.02.009)

The oscillator's natural frequency automatically adapts to match the frequency of an external periodic input signal via a Hebbian learning rule -- no explicit frequency detection is needed.

### Hopf Oscillator: Phase Space, Frequency Adaptation, and Input Tracking

![Hopf oscillator animation](figures/hopf_phasep.gif)

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib, pillow

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── adafreq/                    Oscillator model package
│   ├── hopf.py                 Hopf oscillator
│   ├── vanderpol.py            Van der Pol oscillator
│   ├── rayleigh.py             Rayleigh oscillator
│   ├── rossler.py              Rossler attractor
│   └── fitzhugh_nagumo.py      Fitzhugh-Nagumo oscillator
├── scripts/                    Simulation scripts
│   ├── hopf_simulation.py      Hopf simulations + animated GIF
│   ├── rossler_simulation.py   Rossler simulation + animated GIF
│   ├── rayleigh_simulation.py  Rayleigh simulation
│   ├── vanderpol_simulation.py Van der Pol simulations
│   └── fn_simulation.py        Fitzhugh-Nagumo simulation
├── run_all.py                  Run all simulations
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

### Run all simulations

```bash
python run_all.py
```

### Run a single oscillator

```bash
python scripts/hopf_simulation.py
```

### Use in your own code

```python
import numpy as np
from scipy.integrate import solve_ivp
from adafreq import hopf

t0, tend, Fs = 0, 300, 120
F_t = np.linspace(t0, tend, int(tend * Fs))
F = np.cos(30 * F_t)          # 30 rad/s input

sol = solve_ivp(
    lambda t, x: hopf(t, x, 1, 1, F, F_t),
    [t0, tend], [0, 1, 40], method='RK45', t_eval=F_t
)

import matplotlib.pyplot as plt
plt.plot(sol.t, sol.y[2])     # plot learned frequency
plt.ylabel('W (rad/s)')
plt.show()
```

## Oscillator Models

Each oscillator uses the same Hebbian learning rule structure. The state variables are:

- **X, Y** -- oscillator state (Cartesian coordinates)
- **W** -- learned natural frequency (updated via Hebbian rule)

| Oscillator | Module | Parameters | Notes |
|---|---|---|---|
| Hopf | `adafreq.hopf` | `m` (amplitude), `e` (learning rate) | Classic Hopf normal form |
| Rossler | `adafreq.rossler` | `a`, `b`, `c`, `e` | 4D system (includes Z variable) |
| Rayleigh | `adafreq.rayleigh` | `d`, `q`, `e` | Rayleigh oscillator variant |
| Van der Pol | `adafreq.vanderpol` | `a` (nonlinearity), `e` | Van der Pol with adaptation |
| Fitzhugh-Nagumo | `adafreq.fitzhugh_nagumo` | `a`, `b`, `e` | Neural oscillator model |

## References

- Righetti, L., Buchli, N., & Ijspeert, A. J. (2006). Dynamic Hebbian learning in adaptive frequency oscillators. *Physica D*, 211(3-4), 277-294.
