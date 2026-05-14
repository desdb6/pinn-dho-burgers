"""
Data generation and analytic solution for the damped spring-mass system.

Observation points, collocation points, the initial condition point, and
the validation set are all produced in this script.

Contains:
    analytic           -- analytic solution for y(t)
    make_observations  -- make noisy observation points and split into test and train sets
    make_collocation   -- make collocation points for the physics residual
    generate_data      -- pipeline to generate all necesarry data for model training

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from analytic import gauss_solution_grid, n_wave_solution_grid, step_up_solution_grid, interpolate_solution

def make_observation_gauss(cfg: Config, height: float = 5, sigma: float = 25) -> tuple[np.ndarray, np.ndarray]:
    t_obs = np.random.uniform(0.1, cfg.t_dom, cfg.n_obs)
    x_obs = np.random.uniform(0, cfg.L, cfg.n_obs)
    u_grid, t_arr = gauss_solution_grid(sigma=sigma, height=height, t_end=cfg.t_dom, cfg=cfg)
    u_obs = interpolate_solution(u_grid, t_arr, t_obs, x_obs, cfg)

    return t_obs, x_obs, u_obs

def make_validation(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    t_val= np.linspace(0.1, cfg.t_dom, cfg.n_val)
    y_val = analytic(t_val, cfg) + np.random.normal(0.0, cfg.sigma, cfg.n_val)
    return t_val, y_val

def make_collocation(cfg: Config) -> np.ndarray:
    """
    Generate collocation points.
    """
    t_col_dom = np.linspace(0.1, cfg.t_dom, cfg.n_col_dom)
    t_col_extrap = np.linspace(cfg.t_dom, cfg.t_extrap,
                            int(cfg.n_col_dom * (cfg.t_extrap - cfg.t_dom) / cfg.t_dom))
    return t_col_dom, t_col_extrap

def generate_data(cfg: Config) -> dict:
    """
    Generate all data and return as a dict.
    """
    t_obs, y_obs = make_train_observation(cfg)
    t_val, y_val = make_validation(cfg)
    t_col_dom, t_col_extrap = make_collocation(cfg)

    return {
        "t_obs":        t_obs,
        "y_obs":        y_obs,
        "t_val":        t_val,
        "y_val":        y_val,
        "t_col_dom":    t_col_dom,
        "t_col_extrap": t_col_extrap,
        "t_ic":         np.array([0.0]),
    }

if __name__ == "__main__":
    cfg = Config()
    print(make_observation_gauss(cfg))