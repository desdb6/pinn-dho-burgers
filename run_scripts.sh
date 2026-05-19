#!/bin/bash
# Run the full damped oscillator pipeline:
#   1. Hyperparameter tuning  (tune.py)
#   2. Retrain best models    (train_tuned.py)

set -e  # stop on first error

PYTHON="/home/des/miniconda3/envs/mypythonenv/bin/python"
PROJECT="Damped_Oscillator"

echo ""
echo "============================================================"
echo "  Zeta and omega estimation"
echo "============================================================"
$PYTHON $PROJECT/tune.py
echo "Tuning complete."
$PYTHON $PROJECT/tune_plots.py
echo "Retraining complete."


echo ""
echo "============================================================"
echo "  Zeta and omega estimation"
echo "============================================================"
$PYTHON $PROJECT/parameter_estimation_data_generation.py
echo "Data generation complete."