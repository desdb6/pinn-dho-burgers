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
from plot import plot_summary, plot_epoch_figure
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")

    cfg = Config(
        n_epochs        = 10,
        log_every       = 10,
        patience        = 999,
        snapshot_epochs = (1, 100, 200, 300, 400, 500),
        strata_splitting = 5,
        test_train_split = 0.2
    )
    print(f"Config : {cfg}")

    data     = generate_data(cfg)
    model    = FCNet(cfg)
    history, snapshots = train(
        model       = model,
        data        = data,
        cfg         = cfg,
        device      = device,
        label       = "Model"
    )

    # -- quick sanity checks ------------------------------------------------
    assert len(history["epoch"])     > 0, "history is empty"
    assert len(snapshots)            > 0, "no snapshots saved"
    assert history["loss_total"][-1] > 0, "loss is zero"

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
        device=device
        )
    
    plot_epoch_figure(
        data=data,
        cfg=cfg,
        snapshots=snapshots,
        t_plot_full=t_plot,
        )


if __name__ == "__main__":
    main()