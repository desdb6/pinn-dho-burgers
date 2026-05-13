"""
Training loop for the damped spring-mass PINN.

Contains:
    train -- run the Adam optimiser with cosine-annealing scheduler,
             early stopping, periodic snapshots, and validation logging

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 12/05/2026
"""
import time
import os
from config import Config
import torch
import torch.nn as nn
from utils import to_tensor
from losses import loss_data, loss_ic, loss_physics

os.makedirs("outputs", exist_ok=True)

def train(
    model: nn.Module,
    data: dict,
    cfg: Config,
    device: torch.device,
    label: str = "Model"
) -> tuple[dict, dict]:
    """
    Train the PINN or standard ML model.

    Parameters
    ----------
    model : nn.Module
        The neural network to train. Must already be instantiated on CPU;
        this function moves it to device.
    data : dict
        Output of generate_data(). Must contain keys:
        t_train, y_train, t_val, y_val, t_col, t_ic.
    cfg : Config
        All hyperparameters, physical constants, and loss terms.
    device : torch.device
        Device to train on.
    label : str
        Label used in printed output.

    Returns
    -------
    history : dict
        Loss values logged every cfg.log_every epochs. Keys:
        epoch, loss_data, loss_phys,
        loss_ic, loss_val, loss_total.
    snapshots : dict
        Maps epoch number to CPU state_dict at cfg.snapshot_epochs.

    Saves
    -----
    best_model.pt : 
        The weights that correspond to the lowest validation loss
    """
    # -- move model and tensors to device ------------------------------------
    model.to(device)

    t_train_t       = to_tensor(data["t_train"])
    y_train_t       = to_tensor(data["y_train"])
    t_val_t         = to_tensor(data["t_val"])
    y_val_t         = to_tensor(data["y_val"])
    t_col_dom_t     = to_tensor(data["t_col_dom"], requires_grad=True)
    t_col_extrap_t  = to_tensor(data["t_col_extrap"], requires_grad=True)
    t_ic_t          = to_tensor(data["t_ic"], requires_grad=True)

    # -- initialize optimiser and scheduler --------------------------------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas = (cfg.adam_beta1, cfg.adam_beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    # -- history and snapshot storage ---------------------------------------
    history = {
        "epoch":            [],
        "loss_data":        [],
        "loss_phys":        [],
        "loss_ic":          [],
        "loss_val":         [],
        "loss_total":       [],
    }
    snapshots             = {}
    best_val_loss         = float("inf")
    epochs_no_improvement = 0

    # -- training loop ------------------------------------------------------
    t0 = time.perf_counter()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        # -- data loss ------------------------------------
        y_pred = model(t_train_t)
        l_data = loss_data(y_pred, y_train_t)

        if cfg.use_physics:
            if cfg.train_extrap:
                l_phys = loss_physics(cfg, model, torch.cat([t_col_dom_t, t_col_extrap_t]))
            else:
                l_phys = loss_physics(cfg, model, t_col_dom_t)
        else:
            l_phys = torch.zeros(1, device=device)
            
        l_ic   = loss_ic(cfg, model, t_ic_t) if cfg.use_ic else torch.zeros(1, device=device)

        loss_total   = (l_data + cfg.lambda_phys * l_phys + cfg.lambda_ic   * l_ic)

        # -- update weights ------------------------------------
        loss_total.backward()
        optimiser.step()
        scheduler.step()

        # -- validation ------------------------------------
        model.eval()
        with torch.no_grad():
            y_val_pred = model(t_val_t)
            l_val      = loss_data(y_val_pred, y_val_t)

        # -- early stopping -------------------------------------------------
        if l_val.item() < best_val_loss:
            best_val_loss         = l_val.item()
            epochs_no_improvement = 0
            best_state            = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, "outputs/best_model.pt")
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= cfg.patience:
            print(f"  [{label}] early stopping at epoch {epoch}, improvement stalled.")
            break

        # -- snapshots ------------------------------------------------------
        if epoch in cfg.snapshot_epochs:
            snapshots[epoch] = {
                k: v.cpu() for k, v in model.state_dict().items()
            }

        # -- logging --------------------------------------------------------
        if epoch % cfg.log_every == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(f"  [{label}] epoch {epoch:5d} | "
                  f"L_data={l_data.item():.5f}  "
                  f"L_phys={l_phys.item():.5f}  "
                  f"L_ic={l_ic.item():.5f}  "
                  f"L_val={l_val.item():.5f}  "
                  f"L_total={loss_total.item():.5f}  "
                  f"({elapsed:.1f}s)")

        if epoch % cfg.log_every == 0:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_phys"].append(l_phys.item())
            history["loss_ic"].append(l_ic.item())
            history["loss_val"].append(l_val.item())
            history["loss_total"].append(loss_total.item())

    total_time = time.perf_counter() - t0
    print(f"  [{label}] finished in {total_time:.1f}s "
          f"({total_time / epoch * 1000:.2f} ms/epoch)")

    return history, snapshots