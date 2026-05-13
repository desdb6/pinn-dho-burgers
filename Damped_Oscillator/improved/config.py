"""
Configuration for the damped spring-mass PINN.

All physical constants, hyperparameters, and runtime settings 
for the PINN are defined as a single config class. 
Import and instantiate Config in every script that needs these values.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Physical parameters
    m: float = 1.0       # mass  [kg]
    c: float = 0.1       # damping coefficient
    k: float = 4.0       # spring stiffness  [N/m]

    @property
    def omega_0(self) -> float:
        return float(np.sqrt(self.k/self.m))
    
    @property
    def zeta(self) -> float:
        return float(self.c/(2*np.sqrt(self.m*self.c)))
    
    @property
    def omega_d(self) -> float:
        return float(self.omega_0*np.sqrt(max(0.0, 1.0-self.zeta**2)))

    # Initial conditions
    y0: float = 1.0                 # initial displacement
    dy0: float = 0.0                # initial velocity

    # Time domain
    t_dom: float = 6.0              # End time of known domain
    t_extrap: float = 10.0          # End time of extrapolated domain

    # Loss weights
    use_physics: bool = True
    lambda_phys: float = 1e-2
    use_ic: bool = True
    lambda_ic: float = 1e1
    train_extrap: bool = True

    # Number of data points
    n_obs: float = 50                # Noisy observation points
    strata_splitting: int = 0        # Strata for stratified splitting
    test_train_split: float = 0.2    # Fraction of train set(t)
    n_col_dom: float = 100           # ODE residual collocation points
    sigma: float = 0.05              # Standard deviation for n_obs

    # Network architecture
    hidden: int = 32
    n_layers: int = 4

    # Hyperparameters
    n_epochs: int = 8000            # Number of epochs
    lr: float = 1e-3                # Starting learning rate
    patience: int = 500             # Patience for model stop
    dropout_rate: float = 0         # Dropout rate for every layer
    adam_beta1: float = 0.9         # Adam optimiser: moment hyperparameter
    adam_beta2: float = 0.999       # Adam optimiser: RMSprop hyperparameter
    scheduler_gamma: float = 0.5    # StepRL: scheduler decay factor
    scheduler_step: int = 3000      # StepRL: decay after this many epochs

    # Epoch snapshots
    snapshot_epochs: tuple = (1, 50, 200, 500, 1000, 2000, 4000, 8000)
    log_every: int = 100

    # Paths