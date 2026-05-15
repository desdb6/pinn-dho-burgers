"""
Configuration for the Burger's equation PINN.

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
    nu: float = 0.01                   # viscosity

    # Initial conditions
    ic: str = "Gauss"              # Initial condition type: "Gauss", "N_wave", "N_wave_chop", "Step_up", "Slope"
    height: float = None
    sigma_ic: float = None
    width: float = None
    bc_left: float = None
    bc_right: float = None

    # Time domain
    t_dom: float = 5.0              # End time of known domain
    t_extrap: float = 7.0           # End time of extrapolated domain

    # Space domain
    n_x: int = 1000                     # Spatial resolution for numerical solution
    delta_t = 1e-4                      # Time step for numerical solution if adaptive stepping is disabled
    L: float = 5                        # System length

    @property
    def delta_x(self) -> float:
        return self.L/self.n_x

    # Loss weights
    use_physics: bool = True
    lambda_phys: float = 1e-1
    use_ic: bool = True
    lambda_ic: float = 5e1
    use_bc: bool = True
    lambda_bc: float = 1e1
    train_extrap: bool = True

    # Data generation parameters
    n_obs: float = 200                     # Noisy observation points
    randomise_observation: bool = True    # Randomise observation points every epoch
    n_col_dom: float = 200                # ODE residual collocation points
    sigma: float = 0.01                   # Standard deviation for n_obs
    randomise_collocation: bool = True    # Randomise collocation points every epoch
    n_bc: float = 50                      # Boundary condition points
    randomise_bc_points: bool = True      # Randomise boundary condition loss points every epoch
    n_grid_val: float = 100               # Validation grid size

    # Network architecture
    hidden: int = 32
    n_layers: int = 6

    # Hyperparameters
    n_epochs: int = 30000               # Number of epochs
    lr: float = 1e-3                    # Starting learning rate
    lr_inverse: float = 1e-2            # Learning rate for inverse problem; parameter estimation  
    patience: int = 500                 # Patience for model stop
    patience_thershold: float = 1e-4    # Patience threshold for new best model
    dropout_rate: float = 0.0           # Dropout rate for every layer
    adam_beta1: float = 0.9             # Adam optimiser: moment hyperparameter
    adam_beta2: float = 0.999           # Adam optimiser: RMSprop hyperparameter
    scheduler_gamma: float = 0.5        # StepRL: scheduler decay factor
    scheduler_step: int = 3000          # StepRL: decay after this many epochs

    # Epoch snapshots
    snapshot_epochs: tuple = (1, 50, 300, 1000, 2000, 4000, 10000, 20000)
    log_every: int = 50

    # IC specific parameters
    def __post_init__(self):
        if self.ic == "Gauss":
            self.height     = self.height     or 1.0
            self.sigma_ic   = self.sigma_ic   or 0.5
            self.bc_left    = self.bc_left    or 0.0
            self.bc_right   = self.bc_right   or 0.0

        elif self.ic == "Step_up":
            self.height     = self.height     or 1.0
            self.bc_left    = self.bc_left    or 0.0
            self.bc_right   = self.height     or 1.0

        elif self.ic == "N_wave":
            self.height     = self.height     or 1.0
            self.width      = self.width      or 3.0
            self.bc_left    = self.bc_left    or 0.0
            self.bc_right   = self.bc_right   or 0.0

        elif self.ic == "N_wave_chop":
            self.height     = self.height     or 1.0
            self.width      = self.width      or 3.0
            self.bc_left    = self.bc_left    or 0.0
            self.bc_right   = self.bc_right   or 0.0

        elif self.ic == "Slope":
            self.height     = self.height     or 0.5
            self.width      = self.width      or 3.0
            self.bc_left    = self.height     or 0.5
            self.bc_right   = self.bc_right   or 0.0

        else:
            raise ValueError(f"Unknown IC type: {self.ic}")