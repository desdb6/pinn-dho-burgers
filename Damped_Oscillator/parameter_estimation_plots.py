"""
Script to read a CSV file and plot the predicted-true parameter pairs
and a 2D color coded scatter plot.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from pathlib import Path
from plot import style_ax
from utils import save_show

OUTPUT_PATH = Path.cwd() / "Damped_Oscillator/outputs/parameter_estimation_3"

# -- LaTeX font ------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

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

def parameter_estimation_plot(
        csv_path: Path,
        parameter: str,
        damped_case: str = None,
        output_path: Path = None,
        show: bool = True
) -> None:
    parameter_data = pd.read_csv(csv_path)
    if damped_case is not None:
        parameter_data = parameter_data[parameter_data["damped_case"] == damped_case]

    fig, ax = plt.subplots(figsize=(8, 8))
    style_ax(ax)

    ax.scatter(parameter_data[f"{parameter}_true"], parameter_data[f"{parameter}_pred"], s=5, color = RED)
    max = np.max(parameter_data[f"{parameter}_true"])
    ax.plot(np.linspace(0, max, 2), np.linspace(0, max, 2), color=GRAY, linestyle="--", alpha=0.6)

    damped_case_str = "" if damped_case is None else f"for {damped_case} case"

    if parameter == "zeta":
        ax.set_xlabel(r"$\zeta$", fontsize=20)
        ax.set_ylabel(r"$\hat\zeta$", fontsize=20)
        ax.set_title(rf"$\zeta$ prediction {damped_case_str}", fontsize = 14)
    else:
        ax.set_xlabel(r"$\omega_0$", fontsize=20)
        ax.set_ylabel(r"$\hat\omega_0$", fontsize=20)
        ax.set_title(rf"$\omega_0$ prediction {damped_case_str}", fontsize = 14)

    ax.grid()
    save_show(output_path, show)

def parameter_space_rmse_plot(
    csv_path: Path,
    damped_case: str = None,
    output_path: Path = None,
    show: bool = True
) -> None:
    """
    Scatter plot of every true (zeta, omega_0) pair, color-coded by the
    combined mean squared residual:
        MSE = 0.5 * [(zeta_pred - zeta_true)^2 + (omega_0_pred - omega_0_true)^2]

    A logarithmic color scale is used so both small and large errors are
    visible simultaneously.
    """
    df = pd.read_csv(csv_path)
    if damped_case is not None:
        df = df[df["damped_case"] == damped_case]

    mse = np.sqrt(0.5 * (
        (df["zeta_pred"]    - df["zeta_true"])    ** 2 +
        (df["omega_0_pred"] - df["omega_0_true"]) ** 2
    ))

    fig, ax = plt.subplots(figsize=(8, 7))
    style_ax(ax)

    sc = ax.scatter(
        df["zeta_true"], df["omega_0_true"],
        c=mse, s=18,
        cmap="plasma",
        norm=LogNorm(vmin=mse.clip(lower=1e-10).min(), vmax=mse.max()),
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"RMSE $= \sqrt{\frac{1}{2}\left[(\hat\zeta-\zeta)^2+(\hat\omega_0-\omega_0)^2\right]}$",
                   fontsize=12)
    cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())

    damped_case_str = "" if damped_case is None else f" — {damped_case}"
    ax.set_xlabel(r"$\zeta_{\rm true}$",    fontsize=18)
    ax.set_ylabel(r"$\omega_{0,\rm true}$", fontsize=18)
    ax.set_title(rf"Parameter-space RMSE{damped_case_str}", fontsize=14)
    ax.grid(alpha=0.3)

    save_show(output_path, show)

def main():
    for damped_case in ["overdamped", "underdamped", "critically_damped"]:
        for parameter in ["zeta", "omega_0"]:
            parameter_estimation_plot(
                            csv_path=OUTPUT_PATH / "parameter_pairs.csv",
                            parameter=parameter,
                            damped_case=damped_case,
                            output_path=OUTPUT_PATH / f"scatter_{parameter}_{damped_case}.png",
                            show=False
                            )
            print(f"Saved {damped_case} scatter plot for {parameter}")

    for parameter in ["zeta", "omega_0"]:
            parameter_estimation_plot(
                            csv_path=OUTPUT_PATH / "parameter_pairs.csv",
                            parameter=parameter,
                            output_path=OUTPUT_PATH / f"scatter_{parameter}_full.png",
                            show=False
                            )
            print(f"Saved full scatter plot for {parameter}")

    parameter_space_rmse_plot(
        csv_path=OUTPUT_PATH / "parameter_pairs.csv",
        output_path=OUTPUT_PATH / f"parameter_space_rmse_full.png",
        show=False
    )
    print("Saved parameter space RMSE plot")
if __name__ == "__main__":
     main()