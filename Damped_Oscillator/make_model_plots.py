"""
Save plots from existing model.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from pathlib import Path
from config import Config
from data import generate_data, analytic
from model import FCNet
from plot import save_model_plots
from utils import get_device, load_model

OUTPUT_PATH = Path.cwd() / "Damped_Oscillator/outputs/demo_model"
device = get_device()
model = FCNet(Config())
model, history, snapshots, cfg = load_model(model, OUTPUT_PATH)
data = generate_data(cfg)
save_model_plots(model, history, snapshots, data, cfg, device, OUTPUT_PATH)
print(f"Plots saved to {OUTPUT_PATH}")
