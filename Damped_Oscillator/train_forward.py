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
from trainer import train
from utils import get_device, save_model, load_best_cfg
from plot import save_plots_from_file

OUTPUT_PATH = Path.cwd() / "Damped_Oscillator/outputs/demo_forward_model_bestparams"
OUTPUT_PATH.mkdir(exist_ok=True)

def main():
    device = get_device()
    print(f"Device : {device}")

    cfg = Config()

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