# inr/train.py
from __future__ import annotations
import numpy as np
import torch

def train_mlp(
    lr: float,
    wd_enc: float,
    wd_mlp: float,
    eps: float,
    per_level_scale: float,
    args,
    *,
    splitting_proportion: float = 0.8,
    inference: int = 0,
    splitting: int = 1,
):
    """
    Main training routine.
    Returns a dummy validation loss or a notice string for inference runs.
    """
    if inference:
        return "Training (inference mode) skipped."
    # Pretend we ran for a bit…
    dummy_val = np.random.rand() * 1e-3
    return dummy_val

def objective_function(lr, wd_enc, wd_mlp, eps, per_level_scale):
    """
    Objective for Bayesian optimization.
    """
    class _Args:
        n_steps = 100
        experiment_name = "dummy"
        config = "config.json"
    return -train_mlp(lr, wd_enc, wd_mlp, eps, per_level_scale, _Args())
