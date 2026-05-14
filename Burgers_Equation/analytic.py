"""
Analytic and numerical solution techniques for the Burgers equation.

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import Config

def compute_stable_dt(u: np.ndarray, cfg: Config, cfl: float = 0.4) -> float:
    """
    Compute a stable time step satisfying both the advection CFL condition
    and the diffusion (von Neumann) stability condition.

    CFL (advection) : dt <= cfl * dx / max(|u|)
    Von Neumann (diffusion): dt <= dx^2 / (2 * nu)
    """
    dx = cfg.delta_x
    max_u = np.max(np.abs(u))

    dt_adv  = cfl * dx / max_u if max_u > 1e-12 else np.inf
    dt_diff = 0.4 * dx**2 / cfg.nu   # safety factor 0.4 < 0.5

    return min(dt_adv, dt_diff)

def gauss(height: float, sigma: float, cfg: Config) -> np.ndarray:
    """
    Generate a gauss curve.
    """
    x = np.arange(cfg.n_x)
    mean = cfg.n_x / 2
    return height * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def step_up(height: float, cfg: Config) -> np.ndarray:
    """
    Generate a step-up function.
    """
    u = np.zeros(cfg.n_x)
    mid = cfg.n_x // 2
    u[mid:] = height
    return u

def n_wave(width: float, height: float, cfg: Config) -> np.ndarray:
    """
    Generate an N-wave.
    """
    u = np.zeros(cfg.n_x)

    center = cfg.n_x // 2

    # Positive triangle
    start1 = center - width
    peak1  = center

    # Negative triangle
    peak2  = center
    end2   = center + width

    # Rising positive slope
    u[start1:peak1] = np.linspace(0, height, peak1 - start1)

    # Falling negative slope
    u[peak2:end2] = np.linspace(-height, 0, end2 - peak2)

    return -u

def negative_slope(width: float, height: float, cfg: Config) -> np.ndarray:
    """
    Generate a slope.
    """
    u = np.zeros(cfg.n_x)
    center = cfg.n_x // 2
    width_rel = width // (2 * cfg.delta_x)
    u[:int(center-width_rel)] = height
    u[int(center-width_rel):int(center+width_rel)] = np.linspace(height, 0, int(2 * width_rel))
    return u

def forward_euler(u: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Forward step using Euler method.
    """
    du = np.gradient(u, cfg.delta_x)
    d2u = np.gradient(du, cfg.delta_x)
    u_step = -u * du + cfg.nu * d2u

    return u + cfg.delta_t * u_step

def forward_euler_upwind(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    One forward-Euler step with:
      - First-order upwind differencing for the nonlinear advection term u*du/dx
      - Second-order central differencing for the diffusion term nu*d2u/dx2

    Periodic boundary conditions are used via np.roll.
    """
    dx = cfg.delta_x

    u_right = np.roll(u, -1)   # u[i+1]
    u_left  = np.roll(u,  1)   # u[i-1]

    # Upwind advection: use backward difference where u > 0, forward where u < 0
    adv_pos = u * (u - u_left)  / dx   # u >= 0: backward difference
    adv_neg = u * (u_right - u) / dx   # u <  0: forward  difference

    advection = np.where(u >= 0, adv_pos, adv_neg)

    # Central diffusion
    diffusion = cfg.nu * (u_right - 2*u + u_left) / dx**2

    return u + dt * (-advection + diffusion)

def forward_lax_wendroff(u: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """
    Lax-Wendroff scheme for Burgers' equation.
    Second-order accurate in space and time.
    Diffusion term handled with central differences.
    """
    dx = cfg.delta_x

    u_right = np.roll(u, -1)
    u_left  = np.roll(u,  1)

    # Lax-Wendroff for advection (second-order)
    f       = 0.5 * u**2               # flux f(u) = u²/2
    f_right = 0.5 * u_right**2
    f_left  = 0.5 * u_left**2

    c_right = 0.5 * (u + u_right)      # local wave speed at i+1/2
    c_left  = 0.5 * (u + u_left)       # local wave speed at i-1/2

    flux_right = 0.5*(f + f_right) - 0.5*(dt/dx)*c_right**2*(u_right - u)
    flux_left  = 0.5*(f + f_left)  - 0.5*(dt/dx)*c_left**2 *(u - u_left)

    advection = (flux_right - flux_left) / dx

    # Central diffusion (second-order)
    diffusion = cfg.nu * (u_right - 2*u + u_left) / dx**2

    return u + dt * (-advection + diffusion)

def euler_method(u_init: np.ndarray, t_end: float, cfg: Config,
                 auto_dt: bool = True, cfl: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Burgers' equation using forward Euler + upwind advection.

    Parameters
    ----------
    u_init   : Initial condition u(x, 0).
    t_end    : Final simulation time.
    cfg      : Simulation configuration.
    auto_dt  : If True, recompute a stable dt every step (recommended).
               If False, use cfg.delta_t (make sure it satisfies CFL yourself).
    cfl      : CFL safety factor (default 0.4).

    Returns
    -------
    sol  : np.ndarray of shape (n_snapshots, n_x)  — solution snapshots
    t_arr: np.ndarray of shape (n_snapshots,)       — corresponding times
    """
    u   = u_init.copy()
    t   = 0.0
    sol  = [u.copy()]
    t_arr = [0.0]

    while t < t_end:
        dt = compute_stable_dt(u, cfg, cfl) if auto_dt else cfg.delta_t
        dt = min(dt, t_end - t)
        u  = forward_euler_upwind(u, dt, cfg)
        t += dt

        sol.append(u.copy())
        t_arr.append(t)

    return np.array(sol), np.array(t_arr)

def lax_wendroff(u_init: np.ndarray, t_end: float, cfg: Config,
                 auto_dt: bool = True, cfl: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Burgers' equation using forward Lax Wendroff.

    Parameters
    ----------
    u_init   : Initial condition u(x, 0).
    t_end    : Final simulation time.
    cfg      : Simulation configuration.
    auto_dt  : If True, recompute a stable dt every step (recommended).
               If False, use cfg.delta_t (make sure it satisfies CFL yourself).
    cfl      : CFL safety factor (default 0.4).

    Returns
    -------
    sol  : np.ndarray of shape (n_snapshots, n_x)  — solution snapshots
    t_arr: np.ndarray of shape (n_snapshots,)       — corresponding times
    """
    u   = u_init.copy()
    t   = 0.0
    sol  = [u.copy()]
    t_arr = [0.0]

    while t < t_end:
        dt = compute_stable_dt(u, cfg, cfl) if auto_dt else cfg.delta_t
        dt = min(dt, t_end - t)
        u  = forward_lax_wendroff(u, dt, cfg)
        t += dt

        sol.append(u.copy())
        t_arr.append(t)

    return np.array(sol), np.array(t_arr)

def gauss_solution_grid(sigma: float, height: float, t_end: float, cfg: Config) -> np.ndarray:
    return lax_wendroff(gauss(sigma = sigma, height=height, cfg=cfg), t_end=t_end, cfg=cfg)

def step_up_solution_grid(height: float, t_end: float, cfg: Config) -> np.ndarray:
    return lax_wendroff(step_up(height = height, cfg=cfg), t_end=t_end, cfg=cfg)

def n_wave_solution_grid(width: float, height: float, t_end: float, cfg: Config) -> np.ndarray:
    return lax_wendroff(n_wave(width=width, height=height, cfg=cfg), t_end=t_end, cfg=cfg)

def slope_solution_grid(width: float, height: float, t_end: float, cfg: Config) -> np.ndarray:
    return lax_wendroff(negative_slope(width=width, height=height, cfg=cfg), t_end=t_end, cfg=cfg)

def interpolate_solution(sol: np.ndarray, t_arr: np.ndarray, x: float, t: float, cfg: Config) -> float:
    """
    Bilinear interpolation of the solution at physical coordinates (x, t).
    """
    n_t, n_x = sol.shape

    # Convert physical coords to fractional indices
    xi = x / cfg.delta_x
    ti = np.searchsorted(t_arr, t, side='right') - 1  # handles variable dt

    # Clamp
    xi = np.clip(xi, 0, n_x - 1)
    ti = np.clip(ti, 0, n_t - 2)

    # Integer neighbours
    x0, x1 = int(np.floor(xi)), min(int(np.floor(xi)) + 1, n_x - 1)
    t0, t1 = int(ti), min(int(ti) + 1, n_t - 1)

    # Fractional distances
    dx = xi - x0
    dt = (t - t_arr[t0]) / (t_arr[t1] - t_arr[t0]) if t_arr[t1] != t_arr[t0] else 0.0

    # Corner values
    f00, f10 = sol[t0, x0], sol[t0, x1]
    f01, f11 = sol[t1, x0], sol[t1, x1]

    return (
        f00 * (1 - dx) * (1 - dt) +
        f10 * dx       * (1 - dt) +
        f01 * (1 - dx) * dt +
        f11 * dx       * dt
    )

def residual(sol: np.ndarray, t_arr: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute PDE residual |ut + u*ux - nu*uxx| on the solution grid.
    Uses the time spacing from adaptive stepping.
    """
    dt_arr = np.diff(t_arr)                        # variable spacing
    dt_col = dt_arr[:, np.newaxis]                 # shape (n_t-1, 1) for broadcasting

    # Non-uniform time derivative: forward difference between consecutive steps
    ut = (sol[1:] - sol[:-1]) / dt_col            # shape (n_t-1, n_x)
    u_mid = 0.5 * (sol[1:] + sol[:-1])            # evaluate u at midpoint in time

    ux  = np.gradient(u_mid, cfg.delta_x, axis=1)
    uxx = np.gradient(ux,    cfg.delta_x, axis=1)

    return np.abs(ut + u_mid * ux - cfg.nu * uxx)

def predict_shock_time(u: np.ndarray, cfg: Config) -> float:
    """
    Predict when shockwave will occur using method of characteristics.
    """
    du = np.gradient(u, cfg.delta_x)
    if np.all(du > 0):
        return float('Inf')
    else:
        return -1 / np.min(du)

def plot_method_of_characteristics(u: np.ndarray, cfg: Config, samples: int = 50) -> None:
    """
    Plot the method of characteristics for Burgers' equation.
    Each characteristic is a straight line x(t) = x0 + u0 * t,
    since u is constant along characteristics (inviscid assumption).
    """
    index_samples = np.round(np.linspace(0, cfg.n_x - 1, samples)).astype(int)
    u_samples = u[index_samples]
    x_samples = index_samples * cfg.delta_x

    t_end = cfg.t_dom
    t_line = np.linspace(0, t_end, 200)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (x0, u0) in enumerate(zip(x_samples, u_samples)):
        x_char = x0 + u0 * t_line
        ax.plot(x_char, t_line, lw=1.2, color='blue',
                label="Characteristics" if i == 0 else None)

    shockwave_time = predict_shock_time(u, cfg)
    if shockwave_time is not float("Inf"):
        ax.axhline(shockwave_time, color='red', linestyle='--', label="Shockwave time")

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Method of Characteristics")
    ax.set_xlim(0, cfg.L)
    ax.set_ylim(0, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_anim(sol: np.ndarray, plot_pause: float = 1) -> None:
    """
    Make an animation of the numerical solution.
    """
    n_t, n_x = sol.shape

    x = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(x, sol[0], lw=2)

    ax.set_xlim(0, n_x - 1)
    ax.set_ylim(sol.min(), sol.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Time evolution")

    ax.grid(True, alpha=0.3)

    def update(frame):
        line.set_ydata(sol[frame])
        ax.set_title(f"Time step {frame}/{n_t}")
        return line,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_t,
        interval=plot_pause,
        blit=True
    )

    plt.show()

if __name__ == "__main__":
    cfg = Config()
    # u_init = gauss(height = 5, sigma = 25, cfg=cfg)
    # u_init = n_wave(height = 0.03, width = 100, cfg=cfg)
    # u_init = step_up(height = 1, cfg=cfg)
    u_init = negative_slope(0.4, 0.1, cfg)
    sol, dt_arr = lax_wendroff(u_init, t_end=5, cfg=cfg)
    # plot_anim(sol, 1)
    plot_anim(residual(sol, dt_arr, cfg))