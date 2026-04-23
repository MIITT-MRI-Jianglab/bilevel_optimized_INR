from __future__ import annotations
import argparse, math, os
from bayes_opt import BayesianOptimization

from inr.train import (
                    RunArgs, prepare_data, make_objective,
                    train_mlp, HParams, TrainFlags
)
from inr.load_config import load_config


def _cli():
    p = argparse.ArgumentParser("Bilevel Optimized INR")
    p.add_argument("--mode", choices=["bilevel", "inference"], default="bilevel",
                   help="bilevel: run bilevel optimization, then final inference; inference: one fixed run")

    p.add_argument("--config", type=str, default="inr/config/config_cartesian.json")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--outdir", type=str, default="results/test_run")
    p.add_argument("--experiment_name", type=str, default="test_run")
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--R", type=int, default=6)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--poisson", action="store_true")
    p.add_argument("--vmax-scale", type=float, default=1.0,
                   help="scaling factor for vmax when plotting images")

    p.add_argument("--upper-obj", choices=["self_weighted", "frobenius", "oracle"],
                default="self_weighted", help="upper-level loss")
    p.add_argument("--init_points", type=int, default=20)
    p.add_argument("--n_iters", type=int, default=40)
    p.add_argument("--no-split", action="store_true",
                   help="use no data split for upper-level objective")

    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--wd-enc-min", type=float, default=1e-6)
    p.add_argument("--wd-enc-max", type=float, default=1e-3)
    p.add_argument("--wd-mlp-min", type=float, default=1e-10)
    p.add_argument("--wd-mlp-max", type=float, default=1e-7)
    p.add_argument("--eps-min", type=float, default=1e-5)  # paper notation: delta
    p.add_argument("--eps-max", type=float, default=1e-3)
    p.add_argument("--per-level-scale-min", type=float, default=1.2)
    p.add_argument("--per-level-scale-max", type=float, default=1.6)

    p.add_argument("--lr", type=float, help="inference: learning rate")
    p.add_argument("--wd-enc", type=float, help="inference: encoder weight decay")
    p.add_argument("--wd-mlp", type=float, help="inference: MLP weight decay")
    p.add_argument("--eps", type=float, help="inference: self-weighting epsilon")
    p.add_argument("--per-level-scale", type=float, help="inference: encoding per-level scale")

    return p.parse_args()


def main():
    args = _cli()
    os.makedirs(args.outdir, exist_ok=True)
    _ = load_config(args.config)

    run_args = RunArgs(
        config=args.config,
        data_dir=args.data_dir,
        outdir=args.outdir,
        upper_obj=args.upper_obj,
        n_steps=args.n_steps,
        ndim=args.ndim,
        R=args.R,
        log_every=args.log_every,
        split=args.split,
        cartesian=True,
        poisson=bool(args.poisson),
        device=args.device,
        vmax_scale=args.vmax_scale,
    )
    bundle = prepare_data(run_args)

    if args.mode == "inference":
        missing = [k for k in ["lr", "wd_enc", "wd_mlp", "eps", "per_level_scale"] if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"--mode inference requires: {', '.join('--'+m.replace('_','-') for m in missing)}")

        h = HParams(
            lr=args.lr,
            wd_enc=args.wd_enc,
            wd_mlp=args.wd_mlp,
            eps=args.eps,
            per_level_scale=args.per_level_scale,
        )
        flags = TrainFlags(bilevel=False, oracle=False)
        msg = train_mlp(
            h=h,
            args=run_args,
            bundle=bundle,
            inference=1,
            splitting=0,
            flags=flags,
        )
        print(msg)
        return

    flags = TrainFlags(
        bilevel=True,
        oracle=(args.upper_obj == "oracle"),
    )

    pbounds = {
        "log_lr":          (math.log10(args.lr_min),     math.log10(args.lr_max)),
        "log_wd_enc":      (math.log10(args.wd_enc_min), math.log10(args.wd_enc_max)),
        "log_wd_mlp":      (math.log10(args.wd_mlp_min), math.log10(args.wd_mlp_max)),
        "log_eps":         (math.log10(args.eps_min),    math.log10(args.eps_max)),
        "per_level_scale": (args.per_level_scale_min,    args.per_level_scale_max),
    }

    print(f"[BayesOpt] {len(pbounds)}-D search space, upper-obj='{args.upper_obj}':")
    print(f"  lr              in [{args.lr_min:.1e}, {args.lr_max:.1e}]   (log10 sampling)")
    print(f"  wd_enc          in [{args.wd_enc_min:.1e}, {args.wd_enc_max:.1e}]   (log10 sampling)")
    print(f"  wd_mlp          in [{args.wd_mlp_min:.1e}, {args.wd_mlp_max:.1e}]   (log10 sampling)")
    print(f"  eps             in [{args.eps_min:.1e}, {args.eps_max:.1e}]   (log10 sampling)")
    print(f"  per_level_scale in [{args.per_level_scale_min:.3f}, {args.per_level_scale_max:.3f}]   (linear)")
    print(f"[BayesOpt] init_points={args.init_points}, n_iter={args.n_iters} "
          f"-> {args.init_points + args.n_iters} total trials")

    splitting = 0 if args.no_split else 1
    objective = make_objective(run_args, bundle, flags=flags, splitting=splitting)

    opt = BayesianOptimization(f=objective, pbounds=pbounds, random_state=7, verbose=0)
    opt.maximize(init_points=args.init_points, n_iter=args.n_iters)

    best_log = opt.max["params"]
    best = {
        "lr":              10 ** best_log["log_lr"],
        "wd_enc":          10 ** best_log["log_wd_enc"],
        "wd_mlp":          10 ** best_log["log_wd_mlp"],
        "eps":             10 ** best_log["log_eps"],
        "per_level_scale": best_log["per_level_scale"],
    }
    print("[BayesOpt] best hyper-params →", best)
    print(f"[{args.experiment_name}] best val_loss ≈ {-opt.max['target']:.6g}")

    h_best = HParams(**best)
    msg = train_mlp(h=h_best, args=run_args, bundle=bundle, inference=1, splitting=0, flags=flags)
    print(msg)


if __name__ == "__main__":
    main()