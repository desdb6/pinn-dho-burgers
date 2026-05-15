import random
import numpy as np
import torch
import optuna
import csv
from pathlib import Path
from config import Config
from model import FCNet
from trainer import train
from data import generate_data
from utils import get_device

OUTPUT_PATH = Path.cwd() / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

SEED = 42
CSV_PATH = OUTPUT_PATH / "optuna_results.csv"
CSV_FIELDS = [
    "trial", "value",
    "hidden", "n_layers", "lr", "scheduler_gamma", "scheduler_step",
    "lambda_phys", "lambda_ic", "lambda_bc",
    "final_loss_data", "final_loss_phys", "final_loss_ic",
    "final_loss_bc", "final_loss_val", "final_loss_total",
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def objective(trial: optuna.Trial) -> float:
    set_seed(SEED)

    cfg = Config(
        hidden          = trial.suggest_int("hidden", 16, 128, step=16),
        n_layers        = trial.suggest_int("n_layers", 4, 8),
        lr              = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.3, 0.9),
        scheduler_step  = trial.suggest_int("scheduler_step", 1000, 5000, step=500),
        lambda_phys     = trial.suggest_float("lambda_phys", 1e-3, 1e0, log=True),
        lambda_ic       = trial.suggest_float("lambda_ic", 1e0, 1e2, log=True),
        lambda_bc       = trial.suggest_float("lambda_bc", 1e0, 1e2, log=True),
        n_epochs        = 10000,
        patience        = 2000,
    )

    device = get_device()
    data   = generate_data(cfg)
    model  = FCNet(cfg)
    history, _ = train(model, data, cfg, device, verbatim=False)

    best_val = min(history["loss_val"])

    # -- write row to CSV ---------------------------------------------------
    row = {
        "trial":            trial.number,
        "value":            best_val,
        "hidden":           cfg.hidden,
        "n_layers":         cfg.n_layers,
        "lr":               cfg.lr,
        "scheduler_gamma":  cfg.scheduler_gamma,
        "scheduler_step":   cfg.scheduler_step,
        "lambda_phys":      cfg.lambda_phys,
        "lambda_ic":        cfg.lambda_ic,
        "lambda_bc":        cfg.lambda_bc,
        "final_loss_data":  history["loss_data"][-1],
        "final_loss_phys":  history["loss_phys"][-1],
        "final_loss_ic":    history["loss_ic"][-1],
        "final_loss_bc":    history["loss_bc"][-1],
        "final_loss_val":   history["loss_val"][-1],
        "final_loss_total": history["loss_total"][-1],
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return best_val


Path("outputs").mkdir(exist_ok=True)

study = optuna.create_study(
    direction    = "minimize",
    pruner       = optuna.pruners.MedianPruner(n_warmup_steps=20),
    sampler      = optuna.samplers.TPESampler(seed=SEED),   # fixed sampler seed
    storage      = "sqlite:///optuna.db",
    study_name   = "burgers_pinn",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100, timeout=3600)

print(f"\nBest value : {study.best_value:.6f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k:<25} {v}")