"""
Demo script for training a PINN to solve the inverse 
problem of the damped harmonic oscillator with parameter estimation.

Usage:
    python train_forward.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from pathlib import Path
from config import Config
from data import generate_data, analytic
from model import InverseFCNet, predict
from trainer import train
from plot import plot_summary, plot_epoch_figure
from utils import get_device, convert_to_mck

OUTPUT_PATH = Path.cwd() / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    
    cfg = Config()

    # -- randomise parameters ------------------------------------------------
    # # Underdamped
    # zeta = np.random.uniform(0.1, 0.35)
    # omega_0 = np.random.uniform(1, 5)
    # # Critically damped
    # zeta = 1
    # omega_0 = np.random.uniform(0.5, 5)
    # Overdamped
    zeta = np.random.uniform(1.5, 3)
    omega_0 = np.random.uniform(1, 5)
    m, c, k = convert_to_mck(zeta, omega_0)
    cfg = Config(
        m=m,
        c=c,
        k=k,
        lambda_phys=0.1
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
    t_plot = np.linspace(0.0, cfg.t_extrap, 500)
    y_pred = predict(model, t_plot)

    plot_summary(
        hist=history,
        data=data,
        cfg=cfg,
        y_pinn_full=y_pred,
        t_plot_full=t_plot,
        device=device,
        save_path=OUTPUT_PATH / "backward_summary.png"
        )
    
    # plot_epoch_figure(
    #     data=data,
    #     cfg=cfg,
    #     snapshots=snapshots,
    #     t_plot_full=t_plot,
    #     save_path=OUTPUT_PATH / "forward_epochs.png"
    #     )


if __name__ == "__main__":
    main()