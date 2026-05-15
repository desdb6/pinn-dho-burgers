"""
Plotting functions for the damped spring-mass PINN results.

Contains:
    plot_summary        -- five-panel overview (data, fit, loss, extrapolation,
                           ODE residual)
    plot_epoch_snapshots -- grid of trajectory panels at chosen training epochs

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
import torch.nn as nn
from config import Config
from model import predict, predict_from_state
from utils import rmse
from analytic import interpolate_solution

# -- Color palette ------------------------------------------------
BLUE   = "#378ADD"   # noisy training observations
RED    = "#E24B4A"   # noisy test/validation observations
GREEN  = "#1D9E75"   # PINN
ORANGE = "#EF9F27"   # collocation points
PURPLE = "#7F77DD"   # initial condition marker
GRAY   = "#888780"   # true solution / neutral
LGRAY  = "#D3D1C7"   # spine colour
BG     = "#FAFAF8"   # figure background
PANEL  = "#F1EFE8"   # axes background

def plot_solution_grid(
    model:       nn.Module,
    data:        dict,
    cfg:         Config,
    u_grid:      np.ndarray,
    t_arr:       np.ndarray,
    n_times:     int = 8,
    model_color: str = "#2196F3",
    model_label: str = "PINN",
    fig_title:   str = "PINN — spatial solution at different times",
    clip_y:      bool = False,
    save_path:   Path = Path("outputs/solution_grid.png"),
    show_plot:   bool = True
) -> None:
    """
    Build a 2-column grid of panels, one per time slice.

    Each panel shows:
        - numerical solution u(x, t)     (gray line)
        - model prediction u_hat(x, t)   (coloured line)
        - noisy observations at that t   (blue dots)
        - RMSE annotated in panel title

    Parameters
    ----------
    model       : trained PINN (on CPU)
    data        : output of generate_data()
    cfg         : Config
    u_grid      : numerical solution, shape (n_t, n_x)
    t_arr       : time array from euler_method, shape (n_t,)
    n_times     : number of time slices to plot
    """
    times  = np.linspace(0, cfg.t_extrap, n_times)
    x_plot = np.linspace(0, cfg.L, 300)

    n_cols = 2
    n_rows = (n_times + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True)
    axes_flat = axes.flatten()

    for idx, t in enumerate(times):
        ax = axes_flat[idx]

        # -- numerical solution (interpolated onto x_plot) ------------------
        u_true = np.array([
            interpolate_solution(u_grid, t_arr, x, t, cfg) for x in x_plot
        ])

        # -- model prediction -----------------------------------------------
        t_arr_plot = np.full_like(x_plot, t)
        u_pred = predict(model, t_arr_plot, x_plot)
        if clip_y:
            u_pred = np.clip(u_pred, -3, 3)

        err = np.sqrt(np.mean((u_pred - u_true) ** 2))

        # -- numerical solution ---------------------------------------------
        ax.plot(x_plot, u_true,
                color="#888888", lw=1.4, alpha=0.9, label="Numerical u(x,t)")

        # -- model prediction -----------------------------------------------
        ax.plot(x_plot, u_pred,
                color=model_color, lw=2.0, label=f"{model_label}  û(x,t)")

        # -- observations near this time slice ------------------------------
        dt_window = (cfg.t_extrap) / (2 * n_times)
        mask = np.abs(data["t_obs"] - t) < dt_window
        if mask.any():
            ax.scatter(data["x_obs"][mask], data["u_obs"][mask],
                       color="#2155CD", s=28, marker="o", zorder=5,
                       alpha=0.85, label=f"Observations")

        # -- extrapolation shading ------------------------------------------
        if t > cfg.t_dom:
            ax.set_facecolor("#f5f5f5")

        ax.set_title(f"t = {t:.3f}{'  [extrap]' if t > cfg.t_dom else ''}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.L)
        ax.set_ylabel("u(x, t)", fontsize=8)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x", fontsize=8)

        ax.grid(True, alpha=0.3)

    for idx in range(n_times, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # -- shared legend ------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color="#888888",   lw=1.4, label="Numerical solution"),
        Line2D([0], [0], color=model_color, lw=2.0, label=f"{model_label} prediction"),
        Line2D([0], [0], color="#2155CD", lw=0, marker="o",
               markersize=5, label=f"Observations (N={len(data['t_obs'])})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(fig_title, fontsize=10, y=1.01, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()