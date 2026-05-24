"""
Demo script for training a PINN to solve
the damped harmonic oscillator system.

Usage:
    python train_forward.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

from pathlib import Path

from config import Config
from data import generate_data
from model import FCNet
from plot import save_plots_from_file
from trainer import train
from utils import convert_to_mck, get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    device = get_device()
    print(f"Device : {device}")

    # -- Underdamped ------------------------------------------------
    # output_path = SCRIPT_DIR / "outputs/demo_forward_model_underdamped"
    # output_path.mkdir(exist_ok=True)
    # cfg = Config(m=1.0, c=0.5, k=4.0)

    # -- Critically damped ------------------------------------------
    # output_path = SCRIPT_DIR / "outputs/demo_forward_model_criticallydamped"
    # output_path.mkdir(exist_ok=True)
    # zeta = 1
    # omega_0 = 2
    # m, c, k = convert_to_mck(zeta, omega_0)
    # cfg = Config(m=m, c=c, k=k)

    # -- Overdamped -------------------------------------------------
    output_path = SCRIPT_DIR / "outputs/demo_forward_model_overdamped"
    output_path.mkdir(exist_ok=True)
    zeta = 1.5
    omega_0 = 2
    m, c, k = convert_to_mck(zeta, omega_0)
    cfg = Config(m=m, c=c, k=k)

    # -- train model ------------------------------------------------
    data = generate_data(cfg)
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
    )

    # -- save model -------------------------------------------------
    save_model(best_state, history, snapshots, cfg, output_path)
    print(f"Model saved to {output_path}")

    # -- make plots -------------------------------------------------
    save_plots_from_file(output_path)


if __name__ == "__main__":
    main()
