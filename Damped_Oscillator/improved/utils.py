"""
Utils file for damped spring-mass PINN.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import torch
import numpy as np
from config import Config

def get_device() -> torch.device:
    """Return device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_tensor(arr: np.ndarray, requires_grad: bool = False, unsqueeze: bool = True) -> torch.Tensor:
    """
    Convert a 1-D numpy array to a (N, 1) float32 tensor on DEVICE.
    """
    device = get_device()
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
    if unsqueeze == False:
        t = torch.tensor(arr, dtype=torch.float32).to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate RMSE"""
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def pointwise_residual(y: np.ndarray, t: np.ndarray, cfg: Config):
    dy  = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return np.abs(cfg.m * d2y + cfg.c * dy + cfg.k * y)