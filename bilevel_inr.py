#!/usr/bin/env python3
"""
bilevel_inr.py ― Bilevel optimization for INR training
"""
from __future__ import annotations
import argparse
from bayes_opt import BayesianOptimization

from inr.train import objective_function         
from inr.config import load_config


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Bilevel-INR BayesOpt")
    p.add_argument("--config", default="config.json")
    p.add_argument("--experiment_name", default="dummy-run")
    p.add_argument("--n_steps", type=int, default=2000
    p.add_argument("--init_points", type=int, default=20)
    p.add_argument("--n_iters",     type=int, default=40)
    return p.parse_args()


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    args = _cli()
    cfg = load_config(args.config)         
    print("Loaded config keys:", list(cfg.keys()))

    # Parameter ranges – tweak as you like
    pbounds = {
        "lr":              (1e-4, 1e-2),
        "wd_enc":          (1e-6, 1e-3),
        "wd_mlp":          (1e-9, 1e-6),
        "eps":             (1e-5, 1e-3),
        "per_level_scale": (1.2,  1.6),
    }

    # BayesOpt expects a function that **maximises**.
    # Our pre-made `objective_function` already returns –val_loss,
    opt = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )

    opt.maximize(init_points=args.init_points, n_iter=args.n_iters)

    best = opt.max["params"]
    print("[BayesOpt] best hyper-params →", best)

    # run a last training round with the chosen parameters
    final_val = -opt.max["target"]    
    print(f"[{args.experiment_name}] completed - best val_loss = {final_val}")


if __name__ == "__main__":
    main()