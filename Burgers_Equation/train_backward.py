"""
Demo script for training a PINN to solve 
the damped harmonic oscillator system.

Usage:
    python train_forward.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from pathlib import Path
from config import Config
from data import generate_data
from model import FCNet, InverseFCNet, predict
from trainer import train
from plot import plot_solution_grid
from utils import get_device

OUTPUT_PATH = Path.cwd() / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    # -- randomise viscosity ------------------------------------------------
    # nu = np.random.uniform(0.005, 0.01) # Low viscosity
    nu = np.random.uniform(0.05, 0.1)   # High viscosity
    print("------------------------------------------------"
          f"\nRandomised viscosity : {nu:.4f}"
          "\n------------------------------------------------")
    
    cfg = Config(
        nu=nu,
        hidden=96,
        n_layers=7,
        lr=0.005,
        scheduler_gamma=0.8,
        scheduler_step=2500,
        lambda_phys=0.01,
        lambda_ic=14,
        lambda_bc=40
        )
    
    print(f"Config : {cfg}")

    data     = generate_data(cfg)
    model    = InverseFCNet(cfg)
    history, snapshots = train(
        model       = model,
        data        = data,
        cfg         = cfg,
        device      = device,
        label       = "Model",
    )

    # -- plot ---------------------------------------------------------------
    model.to("cpu")
    plot_solution_grid(model, data, cfg, data["u_grid"], data["t_arr"])


if __name__ == "__main__":
    main()