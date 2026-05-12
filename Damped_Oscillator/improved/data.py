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

def analytic(t: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Closed-form solution for the damped spring-mass system ODE.
    """
    zeta  = cfg.zeta
    omega_0 = cfg.omega_0
    omega_d = cfg.omega_d
    y0 = cfg.y0
    dy0 = cfg.dy0

    if zeta < 1.0:
        #   Underdamped
        #   r = -zeta*omega_0 ± i*omega_d
        #   y(t) = e^{-zeta*omega_0*t} * [A*cos(omega_d*t) + B*sin(omega_d*t)]
        #   A = y0
        #   B = (dy0 + zeta*omega_0*y0) / omega_d
        a  = y0
        b  = (dy0 + zeta * omega_0 * y0) / omega_d

        return np.exp(-zeta * omega_0 * t) * (a * np.cos(omega_d * t) + b * np.sin(omega_d * t))

    elif zeta == 1.0:
        #   Critically damped
        #   r = -omega_0
        #   y(t) = (A + B*t) * e^{-omega_0*t}
        #   A = y0
        #   B = dy0 + omega_0*y0
        a = y0
        b = dy0 + omega_0 * y0

        return (a + b * t) * np.exp(-omega_0 * t)

    else:
        #   Overdamped
        #   r_{1,2} = omega_0 * (-zeta ± sqrt(zeta^2 - 1))
        #   y(t) = A*e^{r1*t} + B*e^{r2*t}
        #   A = (dy0 - r2*y0) / (r1 - r2)
        #   B = (r1*y0 - dy0) / (r1 - r2)
        sqrt_term = np.sqrt(zeta ** 2 - 1.0)
        r1 = omega_0 * (-zeta + sqrt_term)
        r2 = omega_0 * (-zeta - sqrt_term)

        a = (dy0 - r2 * y0) / (r1 - r2)
        b = (r1 * y0 - dy0) / (r1 - r2)

        return a * np.exp(r1 * t) + b * np.exp(r2 * t)

def make_observations(cfg: Config) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """
    Generate noisy observation points and stratify split into test and train sets.
    """
    t = np.random.uniform(0.1, cfg.t_dom, cfg.n_obs)
    y = analytic(t, cfg) + np.random.normal(0.0, cfg.sigma, cfg.n_obs)

    t_obs_train, t_obs_val, y_obs_train, y_obs_val = train_test_split(t, y, test_size=cfg.test_train_split)

    return t_obs_train, y_obs_train, t_obs_val, y_obs_val

def make_collocation(cfg: Config) -> np.ndarray:
    """
    Generate collocation points.
    """
    t_col = np.linspace(0.1, cfg.t_extrap, cfg.n_col)
    return t_col

def generate_data(cfg: Config) -> dict:
    """
    Generate all data and return as a dict.
    """
    t_train, y_train, t_val, y_val = make_observations(cfg)
    t_col = make_collocation(cfg)

    return {
        "t_train": t_train,
        "y_train": y_train,
        "t_val":   t_val,
        "y_val":   y_val,
        "t_col":   t_col,
        "t_ic":    np.array([0.0]),
    }

if __name__ == "__main__":
    my_dict = generate_data(Config())
    print(my_dict["t_train"])