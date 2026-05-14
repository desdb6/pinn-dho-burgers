"""
Loss functions for the damped spring-mass PINN.
Each function takes a model and the relevant tensors
as arguments. 

Contains:
    loss_data    -- MSE between predictions and noisy observations
    loss_physics -- mean squared ODE residual at collocation points
    loss_ic      -- squared error on initial position and velocity

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config

def loss_data(y_pred: torch.Tensor, y_obs_t: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared error between predictions and observations."""
    l_data = torch.mean((y_pred - y_obs_t) ** 2)
    return l_data

def loss_physics(cfg: Config, model: nn.Module, t_col_t: torch.Tensor, x_col_t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points

    r(t) = u_t + u * u_x - nu * u_xx

    All derivatives are obtained via automatic differentiation through the
    network.  create_graph=True on the first call keeps the computation graph
    alive so the second differentiation can flow back through it.

    The [0] unpacks the single-element tuple returned by autograd.grad
    (the API always returns a tuple, one entry per input tensor).
    """
    u_hat = model(t_col_t, x_col_t)

    # first time derivative  du/dt
    u_t = torch.autograd.grad(
        u_hat, t_col_t,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    # first spatial derivative  du/dx
    u_x = torch.autograd.grad(
        u_hat, x_col_t,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    # second spatial derivative  d2u/dx2
    u_xx = torch.autograd.grad(
        u_x, x_col_t,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t + u_hat * u_x - cfg.nu * u_xx
    
    return torch.mean(residual ** 2)

def loss_physics(cfg: Config, model: nn.Module, t_col_t: torch.Tensor, x_col_t: torch.Tensor) -> torch.Tensor:
    """
    L_physics = mean( r(t_i)^2 )  over collocation points

    r(t) = u_t + u * u_x - nu * u_xx

    All derivatives are obtained via automatic differentiation through the
    network.  create_graph=True on the first call keeps the computation graph
    alive so the second differentiation can flow back through it.

    The [0] unpacks the single-element tuple returned by autograd.grad
    (the API always returns a tuple, one entry per input tensor).
    """
    u_hat = model(t_col_t, x_col_t)

    # first time derivative  du/dt
    u_t = torch.autograd.grad(
        u_hat, t_col_t,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    # first spatial derivative  du/dx
    u_x = torch.autograd.grad(
        u_hat, x_col_t,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    # second spatial derivative  d2u/dx2
    u_xx = torch.autograd.grad(
        u_x, x_col_t,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t + u_hat * u_x - model.nu_hat * u_xx
    
    return torch.mean(residual ** 2)

def loss_ic(cfg: Config, model: nn.Module, t_ic_t: torch.Tensor, x_ic_t: torch.Tensor, u_ic_true: torch.Tensor) -> torch.Tensor:
    """
    L_ic = ( u_hat(0) - U0 )^2  +  ( du_hat(0) - DU0 )^2

    Both initial conditions are enforced explicitly.
    u_hat'(0) is obtained via autograd because initial velocity
    is never directly observed in the data.

    LAMBDA_IC >> LAMBDA_PHYS: an error at t=0 compounds over the
    entire trajectory, so this constraint must be tight.
    """
    u_hat_0 = model(t_ic_t, x_ic_t)

    du_0 = torch.autograd.grad(
        u_hat_0, t_ic_t,
        grad_outputs=torch.ones_like(u_hat_0),
        create_graph=True
    )[0]

    return torch.mean((u_hat_0 - u_ic_true) ** 2)