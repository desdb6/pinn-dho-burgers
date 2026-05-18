#!/bin/bash
# Run the full damped oscillator pipeline:
#   1. Hyperparameter tuning  (tune.py)
#   2. Retrain best models    (train_tuned.py)

set -e  # stop on first error

PYTHON="/home/des/miniconda3/envs/mypythonenv/bin/python"

echo ""
echo "============================================================"
echo "  Burgers equation nu estimation"
echo "============================================================"
PROJECT="Burgers_Equation"
$PYTHON $PROJECT/nu_estimation_data_generation.py
echo "Data generation complete."
$PYTHON $PROJECT/nu_estimation_plots.py
echo "Plots made."

echo ""
echo "============================================================"
echo "  Burgers Equation hyperparameter tuning"
echo "============================================================"
PROJECT="Burgers_Equation"
# -- step 1: hyperparameter tuning ---------------------------------------
echo ""
echo "[1/2] Running burgers equation hyperparameter tuning..."
echo "------------------------------------------------------------"
$PYTHON $PROJECT/tune.py
echo "Tuning complete."
$PYTHON $PROJECT/tune_model_plots.py
echo "Retraining complete."

echo ""
echo "============================================================"
echo "  Damped oscillator parameter estimation"
echo "============================================================"
PROJECT="Damped_Oscillator"
$PYTHON $PROJECT/parameter_estimation_data_generation.py
echo "Data generation complete."
$PYTHON $PROJECT/parameter_estimation_plots.py
echo "Plots made."