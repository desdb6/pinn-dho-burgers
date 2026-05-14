"""
Data generation and analytic solution for the Burgers equation.

Observation points, collocation points, the initial conditions, and
the validation set are all produced in this script.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import Config
from analytic import gauss_solution_grid, n_wave_solution_grid, step_up_solution_grid, slope_solution_grid, interpolate_solution

def make_observation(cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_obs = np.random.uniform(0.1, cfg.t_dom, cfg.n_obs)
    x_obs = np.random.uniform(0, cfg.L, cfg.n_obs)

    u_obs = []
    for x, t in zip(x_obs, t_obs):
        u_obs.append(interpolate_solution(u_grid, t_arr, x, t, cfg) + np.random.normal(0.0, cfg.sigma))

    return t_obs, x_obs, np.array(u_obs)

def make_collocation(cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_obs_dom = np.linspace(0.1, cfg.t_dom, cfg.n_col_dom)
    x_obs_dom = np.random.uniform(0, cfg.L, cfg.n_col_dom)

    t_obs_extrap = np.linspace(cfg.t_dom, cfg.t_extrap, int(cfg.n_col_dom * (cfg.t_extrap - cfg.t_dom) / cfg.t_dom))
    x_obs_extrap = np.random.uniform(0, cfg.L, int(cfg.n_col_dom * (cfg.t_extrap - cfg.t_dom) / cfg.t_dom))

    return t_obs_dom, x_obs_dom, t_obs_extrap, x_obs_extrap

def make_validation(cfg: Config, u_grid: np.ndarray, t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_val = np.linspace(0.1, cfg.t_extrap, cfg.n_grid_val)
    x_val = np.linspace(0, cfg.L, cfg.n_grid_val)
    
    u_val = np.zeros((len(x_val), len(t_val)))
    for i, x in enumerate(x_val):
        for j, t in enumerate(t_val):
            u_val[i, j] = interpolate_solution(u_grid, t_arr, x, t, cfg)
    
    return t_val, x_val, u_val

def generate_data(cfg: Config) -> dict:
    """
    Generate all data and return as a dict.
    """
    print("Generating data...")

    if cfg.ic == "Gauss":
        u_grid, t_arr = gauss_solution_grid(cfg)
    elif cfg.ic == "N_wave":
        u_grid, t_arr = n_wave_solution_grid(cfg)
    elif cfg.ic == "Step_up":
        u_grid, t_arr = step_up_solution_grid(cfg)
    elif cfg.ic == "Slope":
        u_grid, t_arr = slope_solution_grid(cfg)

    t_obs, x_obs, u_obs = make_observation(cfg, u_grid, t_arr)
    t_col_dom, x_col_dom, t_col_extrap, x_col_extrap = make_collocation(cfg, u_grid, t_arr)
    t_val, x_val, u_val = make_validation(cfg, u_grid, t_arr)

    return {
        "t_obs":        t_obs,
        "x_obs":        x_obs,
        "u_obs":        u_obs,
        "t_val":        t_val,
        "x_val":        x_val,
        "u_val":        u_val,
        "t_col_dom":    t_col_dom,
        "x_col_dom":    x_col_dom,
        "t_col_extrap": t_col_extrap,
        "x_col_extrap": x_col_extrap,
        "x_arr":        np.linspace(0, cfg.L, u_grid.shape[0]),
        "t_arr":        t_arr, 
        "u_grid":       u_grid,
        "u_ic":         u_grid[0],
        "t_ic":         np.array([0.0]),
    }

def plot_observations_3D(data: dict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data["t_obs"], data["x_obs"], data["u_obs"], color='red', label='Observations')

    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_zlabel('u(t,x)')
    ax.set_title('Observation Points')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    cfg = Config()
    data = generate_data(cfg)    # plot_observations_3D(data)
    plot_observations_3D(data)
