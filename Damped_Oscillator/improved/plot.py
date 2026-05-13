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
import torch
from config import Config
from model import predict, predict_from_state
from utils import pointwise_residual, rmse
from data import analytic

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

def shade_extrap(cfg: Config, ax: Axes) -> None:
    ax.axvspan(cfg.t_dom, cfg.t_extrap, color=GRAY, alpha=0.12)
    ax.axvline(cfg.t_dom, color=GRAY, lw=0.8, ls="--", alpha=0.6)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)

    fig1 = plt.figure(figsize=(15, 12))
    fig1.patch.set_facecolor(BG)
    gs1 = GridSpec(3, 3, figure=fig1,
                            hspace=0.55, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.93,  bottom=0.07)

    ax_data   = fig1.add_subplot(gs1[0, :])
    ax_train  = fig1.add_subplot(gs1[1, :2])
    ax_loss   = fig1.add_subplot(gs1[1, 2])
    ax_extrap = fig1.add_subplot(gs1[2, :2])
    ax_phys   = fig1.add_subplot(gs1[2, 2])

    for ax in [ax_data, ax_train, ax_loss, ax_extrap, ax_phys]:
        style_ax(ax)

def plot_summary(
    hist: dict,
    data: dict,
    cfg: Config,
    y_pinn_full: np.ndarray,
    t_plot_full: np.ndarray,
    device: torch.device,
    save_path: str = "outputs/summary.png"
) -> None:
    """
    Five-panel summary figure for a PINN.

    Panels
    ------
    1. Training data: observations, true trajectory, collocation points
    2. Fit on training interval with RMSE
    3. Training loss curves on log scale
    4. Extrapolation beyond training window
    5. Pointwise ODE residual |r(t)|

    Parameters
    ----------
    hist : dict
        Training history from train() for the PINN.
    data : dict
        Output of generate_data().
    cfg : Config
        Physical constants and domain settings.
    y_pinn_full : np.ndarray
        PINN predictions on t_plot_full.
    y_ml_full : np.ndarray
        ML predictions on t_plot_full.
    t_plot_full : np.ndarray
        Dense time grid over [0, t_extrap].
    device : torch.device
        Device used during training, for display purposes only.
    save_path : str
        Path to save the figure.
    """
    # -- derived quantities -------------------------------------------------
    y_true_full  = analytic(t_plot_full, cfg)
    mask_train   = t_plot_full <= cfg.t_dom
    t_plot_train = t_plot_full[mask_train]
    y_true_train = y_true_full[mask_train]

    rmse_pinn_train = rmse(y_pinn_full[mask_train], y_true_full[mask_train])
    rmse_pinn_ext   = rmse(y_pinn_full[~mask_train], y_true_full[~mask_train])
    phys_res_pinn   = float(np.mean(pointwise_residual(y_pinn_full, t_plot_full, cfg)))

    def _shade_extrap(ax):
        ax.axvspan(cfg.t_dom, cfg.t_extrap, color="gray", alpha=0.10)
        ax.axvline(cfg.t_dom, color="gray", lw=0.8, ls="--", alpha=0.6)

    # -- layout ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_data, ax_train, ax_loss = axes[0]
    ax_extrap, ax_phys, ax_empty = axes[1]
    ax_empty.set_visible(False)

    # -- panel 1: training data --------------------------------------------
    ax_data.set_title(
        "Panel 1 — Training data: sparse noisy observations vs true trajectory\n"
        "t=0 excluded from random observations; y(0) and y'(0) enforced via L_ic",
        fontsize=9, loc="left", pad=6, color="#444441"
    )
    ax_data.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5,
                 label="True trajectory  y(t)")
    ax_data.scatter(data["t_train"], data["y_train"],
                    color=BLUE, s=50, zorder=5, marker="o",
                    label=f"Train observations  (N={int(cfg.n_obs * (1 - cfg.test_train_split))}, sigma={cfg.sigma})")

    ax_data.scatter(data["t_val"], data["y_val"],
                    color=RED, s=50, zorder=5, marker="^",
                    label=f"Validation observations  (N={int(cfg.n_obs * cfg.test_train_split)}, sigma={cfg.sigma})")
    ax_data.scatter([0], [cfg.y0], color=PURPLE, s=120, marker="*", zorder=7,
                    label=f"Initial condition  y(0)={cfg.y0}  [enforced via L_ic]")
    ax_data.scatter(
        data["t_col_dom"][data["t_col_dom"] <= cfg.t_dom],
        np.zeros(np.sum(data["t_col_dom"] <= cfg.t_dom)) - 1.08,
        color=ORANGE, s=8, marker="|", zorder=4, alpha=0.7,
        label="Collocation points (no measurement needed)"
    )
    ax_data.set_xlabel("time  [s]")
    ax_data.set_ylabel("displacement  y(t)")
    ax_data.legend(fontsize=8, framealpha=0.5)
    ax_data.set_xlim(0, cfg.t_dom)

    # -- panel 2: fit on training interval ---------------------------------
    ax_train.set_title(
        f"Panel 2 — Fit on training interval  [0, {cfg.t_dom} s]",
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    ax_train.plot(t_plot_train, y_true_train, color=GRAY, lw=1.5, label="True")
    ax_train.plot(t_plot_train, y_pinn_full[mask_train],
                  color=GREEN, lw=2,
                  label=f"PINN         (RMSE={rmse_pinn_train:.4f})")
    ax_train.scatter(data["t_train"], data["y_train"],
                     color=BLUE, s=30, zorder=5, marker="o", alpha=0.6, label="Train observations")
    ax_train.scatter(data["t_val"], data["y_val"],
                    color=RED, s=50, zorder=5, marker="^", alpha=0.6, label="Test observations")
    ax_train.scatter([0], [cfg.y0], color=PURPLE, s=120, marker="*", zorder=7,
                     label=f"IC  y(0)={cfg.y0}")
    ax_train.set_xlabel("time  [s]")
    ax_train.set_ylabel("displacement  y(t)")
    ax_train.legend(fontsize=7.5, framealpha=0.5)
    ax_train.set_xlim(0, cfg.t_dom)

    # -- panel 3: loss curves ----------------------------------------------
    ax_loss.set_title("Panel 3 — Training loss  (log scale)",
                      fontsize=10, loc="left", pad=6, color="#444441")
    ax_loss.semilogy(hist["epoch"], hist["loss_data"],
                     color=GREEN, lw=1.5,           label="PINN  L_data")
    for ep in cfg.snapshot_epochs[:-1]:
        ax_loss.axvline(ep, color=GRAY, lw=0.5, ls=":", alpha=0.5)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=7, framealpha=0.5)

    # -- panel 4: extrapolation --------------------------------------------
    ax_extrap.set_title(
        f"Panel 4 — Extrapolation beyond training window  [{cfg.t_dom}, {cfg.t_extrap} s]",
        fontsize=10, loc="left", pad=6, color="#444441"
    )
    _shade_extrap(ax_extrap)
    ax_extrap.plot(t_plot_full, y_true_full, color=GRAY, lw=1.5, label="True")
    ax_extrap.plot(t_plot_full, y_pinn_full,
                   color=GREEN, lw=2,
                   label=f"PINN         (extrap RMSE={rmse_pinn_ext:.4f})")
    ax_extrap.scatter(data["t_train"], data["y_train"],
                      color=BLUE, s=30, zorder=5, marker="o", alpha=0.5, label="Observations")
    ax_extrap.scatter(data["t_val"], data["y_val"],
                    color=RED, s=50, zorder=5, marker="^", alpha=0.6, label="Test observations")
    ax_extrap.set_xlabel("time  [s]")
    ax_extrap.set_ylabel("displacement  y(t)")
    ax_extrap.set_ylim(-2.5, 2.5)
    ax_extrap.set_xlim(0, cfg.t_extrap)
    ax_extrap.legend(fontsize=7.5, framealpha=0.5)
    ax_extrap.annotate("training region", xy=(0.05, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)
    ax_extrap.annotate("extrapolation",   xy=(0.72, 0.93),
                       xycoords="axes fraction", fontsize=7.5, color=GRAY)

    # -- panel 5: pointwise ODE residual -----------------------------------
    ax_phys.set_title("Panel 5 — Pointwise ODE residual  |r(t)|",
                      fontsize=10, loc="left", pad=6, color="#444441")
    _shade_extrap(ax_phys)
    r_pinn = pointwise_residual(y_pinn_full, t_plot_full, cfg)
    ax_phys.plot(t_plot_full, r_pinn, color=GREEN, lw=1.5,
                 label=f"PINN  (mean={phys_res_pinn:.3f})")
    ax_phys.set_xlabel("time  [s]")
    ax_phys.set_ylabel("|m*y'' + c*y' + k*y|")
    ax_phys.set_xlim(0, cfg.t_extrap)
    ax_phys.legend(fontsize=7.5, framealpha=0.5)

    # -- suptitle ----------------------------------------------------------
    fig.suptitle(
        f"PINN vs Standard ML  |  device={device}  |  "
        f"m={cfg.m}  c={cfg.c}  k={cfg.k}  "
        f"zeta={cfg.zeta:.3f}  omega_d={cfg.omega_d:.3f} rad/s\n"
        f"{cfg.n_obs} observations (t>0)  sigma={cfg.sigma}  |  "
        f"{cfg.n_col_dom} collocation pts  |  "
        f"lambda_phys={cfg.lambda_phys}  lambda_ic={cfg.lambda_ic}  |  "
        f"IC: y(0)={cfg.y0}  y'(0)={cfg.dy0}",
        fontsize=9, y=0.975, color="#2C2C2A"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()

def plot_epoch_figure(
    data:        dict,
    cfg:         Config,
    snapshots:   dict,
    t_plot_full: np.ndarray,
    model_color: str = GREEN,
    model_label: str = "PINN",
    fig_title:   str = f"PINN — trajectory evolution across training epochs",
    clip_y:      bool = False,
    save_path:   str = "outputs/epoch_figure.png"
) -> None:
    """
    Build a 2-column grid of panels, one per snapshot epoch.

    Each panel shows:
        - true analytic solution         (gray line,    full domain)
        - model prediction at that epoch (coloured line, full domain)
        - noisy train observations       (blue dots,    training window)
        - noisy val observations         (red triangles, training window)
        - initial condition              (purple star,  t=0)
        - collocation points             (orange ticks, full domain)
        - training / extrapolation boundary (dashed vertical line)
        - RMSE over full domain annotated in panel title

    Snapshots were saved as CPU state_dicts, so predict_from_state
    runs entirely on CPU -- no device transfer needed here.

    Parameters
    ----------
    hist : dict
        Training history from train().
    data : dict
        Output of generate_data().
    cfg : Config
        Physical constants and domain settings.
    snapshots : dict
        Maps epoch number to CPU state_dict.
    t_plot_full : np.ndarray
        Dense time grid over [0, t_extrap].
    model_color : str
        Hex colour string for the model prediction line.
    model_label : str
        Short label for the model (e.g. "PINN").
    fig_title : str
        Figure suptitle string.
    filename : str
        Path to save the figure.
    clip_y : bool
        If True, clip predictions to [-3, 3] to avoid blown-up axes.
    """
    epochs  = sorted(snapshots.keys())
    n_snap  = len(epochs)
    n_cols  = 2
    n_rows  = (n_snap + 1) // n_cols

    # -- derived quantities -------------------------------------------------
    y_true_full = analytic(t_plot_full, cfg)
    mask_train  = t_plot_full <= cfg.t_dom

    t_col_full  = np.concatenate([data["t_col_dom"], data["t_col_extrap"]])

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    y_lo, y_hi = -2.2, 1.6
    col_y = y_lo + 0.08 * (y_hi - y_lo)

    for idx, epoch in enumerate(epochs):
        ax = axes_flat[idx]

        y_pred = predict_from_state(snapshots[epoch], t_plot_full, cfg)
        if clip_y:
            y_pred = np.clip(y_pred, -3, 3)

        err = rmse(y_pred, y_true_full)

        # true solution
        ax.plot(t_plot_full, y_true_full,
                color=GRAY, lw=1.4, alpha=0.9, label="True  y(t)")

        # model prediction
        ax.plot(t_plot_full, y_pred,
                color=model_color, lw=2.0, label=f"{model_label}  ŷ(t)")

        # extrapolation shading
        ax.axvspan(cfg.t_dom, cfg.t_extrap, color=GRAY, alpha=0.10, zorder=0)
        ax.axvline(cfg.t_dom, color=GRAY, lw=0.8, ls="--", alpha=0.5)

        # collocation ticks
        ax.scatter(t_col_full, np.full_like(t_col_full, col_y),
                   color=ORANGE, s=6, marker="|", alpha=0.6, zorder=3,
                   label=f"Collocation pts  (N={len(t_col_full)})")

        # train observations
        ax.scatter(data["t_train"], data["y_train"],
                   color=BLUE, s=28, marker="o", zorder=5, alpha=0.85,
                   label=f"Train obs  (N={len(data['t_train'])})")

        # validation observations
        ax.scatter(data["t_val"], data["y_val"],
                   color=RED, s=28, marker="^", zorder=5, alpha=0.85,
                   label=f"Val obs  (N={len(data['t_val'])})")

        # initial condition
        ax.scatter([0], [cfg.y0],
                   color=PURPLE, s=130, marker="*", zorder=7,
                   label=f"IC  y(0)={cfg.y0}")

        ax.set_title(f"epoch = {epoch}   |   RMSE = {err:.4f}",
                     fontsize=9, loc="left", pad=4, color="#444441")
        ax.set_xlim(0, cfg.t_extrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y(t)", fontsize=8)

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time  [s]", fontsize=8)

        if idx == 0:
            ax.text(cfg.t_dom / 2, y_hi - 0.15, "training",
                    ha="center", va="top", fontsize=7, color=GRAY)
            ax.text(cfg.t_dom + (cfg.t_extrap - cfg.t_dom) / 2,
                    y_hi - 0.15, "extrapolation",
                    ha="center", va="top", fontsize=7, color=GRAY)

    for idx in range(n_snap, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # -- shared legend ------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=GRAY,        lw=1.4,
               label="True solution  y(t)"),
        Line2D([0], [0], color=model_color, lw=2.0,
               label=f"{model_label} prediction  ŷ(t)"),
        Line2D([0], [0], color=BLUE,   lw=0, marker="o", markersize=5,
               label=f"Train observations  "
                     f"(N={len(data['t_train'])}, sigma={cfg.sigma})"),
        Line2D([0], [0], color=RED,    lw=0, marker="^", markersize=5,
               label=f"Val observations  "
                     f"(N={len(data['t_val'])}, sigma={cfg.sigma})"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="*", markersize=9,
               label=f"IC  y(0)={cfg.y0},  y'(0)={cfg.dy0}"),
        Line2D([0], [0], color=ORANGE, lw=0, marker="|", markersize=7,
               label=f"Collocation pts  "
                     f"(N={len(t_col_full)}, [0, {cfg.t_extrap}])"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(fig_title, fontsize=10, y=1.01, color="#2C2C2A")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()