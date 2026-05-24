"""
Load the best hyperparameter-tuned model for each regime,
retrain it, and save plots to the correct folders.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""
from pathlib import Path
from data import generate_data
from model import FCNet
from trainer import train
from utils import get_device, save_model, load_best_cfg
from plot import save_plots_from_file

OUTPUT_PATH = Path.cwd() / "Damped_Oscillator/outputs/optuna_tuning_2"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DAMPED_CASES = ["underdamped", "critically", "overdamped"]


def main():
    """Main loop."""
    device = get_device()
    print(f"Device : {device}\n")

    for damped_case in DAMPED_CASES:
        print(f"{'='*60}")
        print(f"Regime: {damped_case}")
        print(f"{'='*60}")

        # -- load best hyperparameters for this damped case --------
        best_params_path = OUTPUT_PATH / damped_case / "best_params.json"
        if not best_params_path.exists():
            print(f"  No best_params.json found for {damped_case} — skipping.")
            continue

        cfg = load_best_cfg(best_params_path)
        print(f"  Best cfg : {cfg}")

        # -- output folder for this damped case --------------------
        damped_case_output = OUTPUT_PATH / damped_case
        damped_case_output.mkdir(parents=True, exist_ok=True)

        # -- train -------------------------------------------------
        data = generate_data(cfg)
        model = FCNet(cfg)
        history, snapshots, best_state = train(
            model=model,
            data=data,
            cfg=cfg,
            device=device,
            label=damped_case,
        )
        # -- save ---------------------------------------------------
        save_model(best_state, history, snapshots, cfg, damped_case_output)
        print(f"  Model saved to {damped_case_output}")

        # -- plots --------------------------------------------------
        save_plots_from_file(damped_case_output)
        print(f"  Plots saved to {damped_case_output}\n")


if __name__ == "__main__":
    main()
