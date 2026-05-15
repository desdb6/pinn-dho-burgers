"""
Optuna hyperparameter search for the damped oscillator PINN.

Usage:
    python tune.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import random
import csv
import json
import numpy as np
import torch
import optuna
from pathlib import Path

from config import Config
from model import FCNet
from trainer import train
from data import generate_data
from utils import get_device

# -- settings --------------------------------------------------------------
SEED        = 42
N_TRIALS    = 200
TIMEOUT     = 3600          # seconds
OUTPUT_PATH = Path.cwd() / "Damped_Oscillator/outputs/optuna"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_PATH / "optuna_results.csv"
CSV_FIELDS = [
    "trial", "state", "value",
    # architecture
    "hidden", "n_layers", "dropout_rate",
    # optimiser
    "lr", "scheduler_gamma", "scheduler_step",
    # loss weights
    "lambda_phys", "lambda_ic",
    # final losses
    "final_loss_data", "final_loss_phys", "final_loss_ic",
    "final_loss_val",  "final_loss_total",
]

# -- fixed seed ------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -- generate data once ----------------------------------------------------
_base_cfg  = Config()
_base_data = generate_data(_base_cfg)

# -- objective -------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    set_seed(SEED)

    cfg = Config(
        # architecture
        hidden        = trial.suggest_int(  "hidden",           16,   128,  step=16),
        n_layers      = trial.suggest_int(  "n_layers",          2,     6),
        dropout_rate  = trial.suggest_float("dropout_rate",     0.0,   0.2),
        # optimiser
        lr            = trial.suggest_float("lr",               1e-4,  1e-2, log=True),
        scheduler_gamma = trial.suggest_float("scheduler_gamma",0.3,   0.9),
        scheduler_step  = trial.suggest_int("scheduler_step",   500,  5000, step=500),
        # loss weights
        lambda_phys   = trial.suggest_float("lambda_phys",      1e-3,  1e1, log=True),
        lambda_ic     = trial.suggest_float("lambda_ic",        1e0,   1e2, log=True),
        # fixed budget during search
        n_epochs      = 10000,
        patience      = 2000,
    )

    device = get_device()
    model  = FCNet(cfg)

    try:
        history, _, _ = train(
            model        = model,
            data         = _base_data,
            cfg          = cfg,
            device       = device,
            verbatim     = False,
            optuna_trial = trial,
        )
    except optuna.exceptions.TrialPruned:
        raise

    best_val = min(history["loss_val"])

    # -- write to CSV ------------------------------------------------------
    row = {
        "trial":             trial.number,
        "state":             "complete",
        "value":             best_val,
        "hidden":            cfg.hidden,
        "n_layers":          cfg.n_layers,
        "dropout_rate":      cfg.dropout_rate,
        "lr":                cfg.lr,
        "scheduler_gamma":   cfg.scheduler_gamma,
        "scheduler_step":    cfg.scheduler_step,
        "lambda_phys":       cfg.lambda_phys,
        "lambda_ic":         cfg.lambda_ic,
        "final_loss_data":   history["loss_data"][-1],
        "final_loss_phys":   history["loss_phys"][-1],
        "final_loss_ic":     history["loss_ic"][-1],
        "final_loss_val":    history["loss_val"][-1],
        "final_loss_total":  history["loss_total"][-1],
    }
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return best_val


# -- callback --------------------------------------------------------------
def print_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    print(
        f"  Trial {trial.number:>4} | "
        f"State: {trial.state.name:<10} | "
        f"Value: {f'{trial.value:.6f}' if trial.value is not None else 'pruned':>12} | "
        f"Best:  {study.best_value:.6f}"
    )


# -- run -------------------------------------------------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        direction     = "minimize",
        pruner        = optuna.pruners.MedianPruner(n_warmup_steps=20),
        sampler       = optuna.samplers.TPESampler(seed=SEED),
        storage       = f"sqlite:///{OUTPUT_PATH}/optuna.db",
        study_name    = "damped_oscillator_pinn",
        load_if_exists= True,
    )

    print(f"Starting study — {N_TRIALS} trials, timeout {TIMEOUT}s")
    print(f"Results: {CSV_PATH}")
    print(f"DB:      {OUTPUT_PATH}/optuna.db\n")

    study.optimize(
        objective,
        n_trials   = N_TRIALS,
        timeout    = TIMEOUT,
        callbacks  = [print_callback],
    )

    # -- summary -----------------------------------------------------------
    print(f"\nBest value : {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:<25} {v}")

    # -- save best params as JSON ------------------------------------------
    with open(OUTPUT_PATH / "best_params.json", "w") as f:
        json.dump({
            "best_value":  study.best_value,
            "best_params": study.best_params,
        }, f, indent=4)

    # -- save full trial table ---------------------------------------------
    df = study.trials_dataframe()
    df.to_csv(OUTPUT_PATH / "all_trials.csv", index=False)
    print(f"\nFull trial table saved to {OUTPUT_PATH}/all_trials.csv")