"""
Smoke test for the damped spring-mass PINN pipeline.

Runs a short training session with reduced epochs to verify that all
modules import correctly, data generation works, the model trains
without errors, and outputs are saved correctly.

Usage:
    python test_pipeline.py
    python test_pipeline.py --cpu

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data import generate_data
from model import FCNet, predict
from trainer import train
from utils import get_device

import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data import generate_data, analytic
from model import FCNet, predict
from trainer import train
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")

    cfg = Config(
        n_epochs        = 500,
        log_every       = 10,
        patience        = 999,
        snapshot_epochs = (1, 25, 50),
    )
    print(f"Config : {cfg}")

    data     = generate_data(cfg)
    model    = FCNet(cfg)
    history, snapshots = train(
        model       = model,
        data        = data,
        cfg         = cfg,
        device      = device,
        use_physics = True,
        label       = "smoke test"
    )

    # -- quick sanity checks ------------------------------------------------
    assert len(history["epoch"])     > 0, "history is empty"
    assert len(snapshots)            > 0, "no snapshots saved"
    assert history["loss_total"][-1] > 0, "loss is zero"

    # -- plot ---------------------------------------------------------------
    model.to("cpu")
    t_plot = np.linspace(0.0, cfg.t_extrap, 500)
    y_true = analytic(t_plot, cfg)
    y_pred = predict(model, t_plot)

    plt.figure(figsize=(10, 4))
    plt.axvspan(cfg.t_dom, cfg.t_extrap, color="gray", alpha=0.15,
                label="Extrapolation")
    plt.axvline(cfg.t_dom, color="gray", lw=0.8, ls="--")
    plt.plot(t_plot, y_true, color="gray", lw=1.5, label="Analytic")
    plt.plot(t_plot, y_pred, color="green", lw=2.0, label="PINN")
    plt.scatter(data["t_train"], data["y_train"],
                color="blue", s=30, zorder=5, label="Train")
    plt.scatter(data["t_val"], data["y_val"],
                color="red", s=30, zorder=5, marker="^", label="Val")
    plt.scatter([0.0], [cfg.y0],
                color="purple", s=150, marker="*", zorder=7, label="IC")
    plt.xlabel("t [s]")
    plt.ylabel("y(t)")
    plt.title("Smoke test — PINN vs analytic")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/smoke_test.png", dpi=150)
    plt.show()
    print("All checks passed.")


if __name__ == "__main__":
    main()