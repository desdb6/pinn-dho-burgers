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
from model import FCNet, predict
from trainer import train
from plot import plot_solution_grid
from utils import get_device

OUTPUT_PATH = Path.cwd() / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    cfg = Config()
    print(f"Config : {cfg}")

    data     = generate_data(cfg)
    model    = FCNet(cfg)
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