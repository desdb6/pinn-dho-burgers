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
from analytic import predict_shock_time
from data import generate_data
from model import FCNet
from trainer import train
from plot import save_plots_from_file
from utils import get_device, save_model

OUTPUT_PATH = Path.cwd() / "Burgers_Equation/outputs/forward_demo_gauss"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    cfg = Config(
        ic="Gauss"
    )

    time_to_shock = predict_shock_time(cfg)
    if time_to_shock <= 5 and time_to_shock >= 0.1:
        print("--------------------------------------------------"
              "\n"
              f"Shockwave predicted to occur at {time_to_shock:.3f}s."
              "\n"
              "Setting training domain to end at shock time."
              "\n"
              "--------------------------------------------------")
        cfg.t_shock = time_to_shock
        cfg.t_dom = time_to_shock + 2.0
        cfg.t_extrap = time_to_shock + 4.0

    # -- train model ------------------------------------------------
    data     = generate_data(cfg)
    model    = FCNet(cfg)
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