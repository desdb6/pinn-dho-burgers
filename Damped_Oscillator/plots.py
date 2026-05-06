"""
Plots fot the discussion on the Physics-Informed Neural Network (PINN) for a Damped Spring-Mass System
Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 06/05/26
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pinn_spring_damper_des import FCNet, train, analytic, predict, SNAPSHOT_EPOCHS
import pinn_spring_damper_des as PSN

# ── Plot settings ──────────────────────────────────────────────────────────
DPI=360

BLUE   = "#378ADD"   # noisy observations
RED    = "#E24B4A"   # standard ML
GREEN  = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spine colour
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background

def initial_condition_loss_model():
    model_no_ic = FCNet(hidden=32, n_layers=4)
    hist_no_ic, snaps_no_ic = train(
        model_no_ic, use_physics=True, use_ic=False,
        label="NO IC LOSS", snapshot_epochs=SNAPSHOT_EPOCHS,
        save_path='models/1.1.2_no_ic.pth'
        )

    model_ic = FCNet(hidden=32, n_layers=4)
    hist_ic, snaps_ic = train(
        model_ic, use_physics=True, use_ic=True,
        label="WITH IC LOSS", snapshot_epochs=SNAPSHOT_EPOCHS,
        save_path='models/1.1.2_ic.pth'
    )

def initial_condition_loss_plot():
    """Compare the results of the model with or without the inclusion of initial condition loss."""
    t = np.linspace(0, 10, 1000)
    y_gt = analytic(t)

    model_no_ic = FCNet(hidden=32, n_layers=4)
    model_no_ic.load_state_dict(torch.load("Damped_Oscillator/models/1.1.2_no_ic.pth", weights_only=True))
    y_no_ic = predict(model_no_ic, t)

    model_ic = FCNet(hidden=32, n_layers=4)
    model_ic.load_state_dict(torch.load("Damped_Oscillator/models/1.1.2_ic.pth", weights_only=True))
    y_ic = predict(model_ic, t)

    fig, ax = plt.subplots(figsize=(16,8))

    PSN.shade_extrap(ax)
    PSN.style_ax(ax)

    ax.plot(t, y_gt, label='Ground truth', color=GRAY)
    ax.plot(t, y_no_ic, label='No $\mathcal{L}_{i.c.}$', color=RED)
    ax.plot(t, y_ic, label='With $\mathcal{L}_{i.c.}$', color=GREEN)
    ax.scatter(PSN.t_obs, PSN.y_obs,
                color=BLUE, s=28, zorder=5, alpha=0.85,
                label=f"Observations")

    ax.legend()
    ax.grid()
    ax.set_title("Effect of $\mathcal{L}_{i.c.}$ in Damped Oscillator PINN", fontsize=16)
    ax.set_xlabel("time  [s]", fontsize=12)
    ax.set_ylabel("y(t)", fontsize=12)

    plt.savefig("Damped_Oscillator/plots/ic_effect.png", dpi=DPI)
    print("Saved ic loss effect plot at Damped_Oscillator/plots/ic_effect.png")
    plt.close()

def main():
    """Make and save all plots."""
    initial_condition_loss_plot()

if __name__ == "__main__":
    main()

    