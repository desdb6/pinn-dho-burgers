#!/bin/bash
# Run the full damped oscillator pipeline:
#   1. Hyperparameter tuning  (tune.py)
#   2. Retrain best models    (train_tuned.py)

set -e  # stop on first error

PYTHON="/home/des/miniconda3/envs/mypythonenv/bin/python"
PROJECT="Burgers_Equation"

$PYTHON $PROJECT/train_forward_demonstration_cases.py
$PYTHON $PROJECT/train_forward_data_included.py
$PYTHON $PROJECT/train_forward_extrapblind.py
$PYTHON $PROJECT/train_forward_PINNvsML.py
$PYTHON $PROJECT/train_forward_hyperparameters.py
# $PYTHON $PROJECT/nu_estimation_data_generation.py