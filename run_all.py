"""Run All Simulations

Executes all five adaptive-frequency oscillator simulations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from scripts import hopf_simulation, rossler_simulation, rayleigh_simulation
from scripts import vanderpol_simulation, fn_simulation

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    print('Running Hopf oscillator simulation...')
    hopf_simulation.run()

    print('Running Rossler oscillator simulation...')
    rossler_simulation.run()

    print('Running Rayleigh oscillator simulation...')
    rayleigh_simulation.run()

    print('Running Van der Pol oscillator simulation...')
    vanderpol_simulation.run()

    print('Running Fitzhugh-Nagumo oscillator simulation...')
    fn_simulation.run()

    print('All simulations complete.')
