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
from plot import save_plots_from_file
from utils import get_device, save_model

OUTPUT_PATH = Path.cwd() / "Burgers_Equation/outputs/backward_demo_N_wave_highnu"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    # -- randomise viscosity ------------------------------------------------
    # nu = np.random.uniform(0.005, 0.015) # Low viscosity
    nu = np.random.uniform(0.1, 0.2)   # High viscosity
    print("------------------------------------------------"
          f"\nRandomised viscosity : {nu:.4f}"
          "\n------------------------------------------------")

    cfg = Config(
        ic="N_wave",
        nu=nu
    )

    print(f"Config : {cfg}")

    data     = generate_data(cfg)
    model    = InverseFCNet(cfg)
    history, snapshots, best_state = train(
        model       = model,
        data        = data,
        cfg         = cfg,
        device      = device,
        label       = "Model",
    )

    # -- save model ------------------------------------------------
    save_model(best_state, history, snapshots, cfg, OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")

    # -- make plots ------------------------------------------------
    save_plots_from_file(OUTPUT_PATH)


if __name__ == "__main__":
    main()