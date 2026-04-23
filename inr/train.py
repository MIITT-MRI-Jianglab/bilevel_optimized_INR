from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, Literal, Union
import numpy as np
import torch
import os

from .load_config import load_config
from .data import load_kspace
from .loss_fn import self_weighted_l2_loss
from .model import create_model
from .utils.utils import (
    create_meshgrid, fftnd_torch, coil_combine, img_evaluation,
    plot_image, plot_images_multi, compute_grid,
    separate_mask, cartesian_undersampling,
)


@dataclass
class RunArgs:
    config: str
    data_dir: str
    outdir: str
    upper_obj: Literal["self_weighted", "frobenius", "oracle"] = "self_weighted"
    n_steps: int = 2000
    ndim: int = 2                # 2 or 3
    R: int = 4                   # undersampling ratio
    log_every: int = 50
    split: float = 0.8           # train/val split for bilevel optim
    cartesian: bool = True
    poisson: bool = False
    device: str = "cuda"
    vmax_scale: float = 1.0


@dataclass(frozen=True)
class HParams:
    lr: float
    wd_enc: float
    wd_mlp: float
    eps: float
    per_level_scale: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainFlags:
    bilevel: bool = True
    oracle: bool = False   # image-domain oracle upper-level loss


@dataclass
class DataBundle:
    kdata: torch.Tensor           # (coil, ...) complex64, on device
    b1: torch.Tensor              # (coil, ...) complex64, on device
    matrix_size: Tuple[int, ...]  # (ny,nx) or (nz,ny,nx)
    fft_axes: Tuple[int, ...]
    ncoil: int
    image_ref_np: np.ndarray      # coil-combined reference (CPU)
    mask_np: np.ndarray           # ROI mask from b1 (CPU)
    coords: torch.Tensor          # (N, D) float32 on device
    us_mask: torch.Tensor         # training mask
    val_mask: torch.Tensor        # validation mask (unused in oracle mode)
    full_mask: torch.Tensor       # training+validation union


def prepare_data(args: RunArgs) -> DataBundle:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    kdata_t, b1_t = load_kspace(args.data_dir)
    kdata_t = kdata_t.to(torch.complex64).to(device)
    b1_t    = b1_t.to(torch.complex64).to(device)

    kdata_t = kdata_t / (torch.max(torch.abs(kdata_t)) + 1e-12)

    if args.ndim == 2:
        matrix_size = tuple(kdata_t.shape[-2:])
        fft_axes = (-2, -1)
    else:
        matrix_size = tuple(kdata_t.shape[-3:])
        fft_axes = (-3, -2, -1)
    ncoil = int(kdata_t.shape[0])

    with torch.no_grad():
        image_ref = coil_combine(
            kdata_t, b1_t, domain="kspace", mode="sense",
            coil_axis=0, fft_ndims=args.ndim
        )
    image_ref_np = np.squeeze(image_ref.detach().cpu().numpy())
    plot_image(np.abs(image_ref_np),
                f"{args.outdir}/ref_img.png",
                "Fully sampled Recon",
                dpi=200, vmax_scale=args.vmax_scale)
    mask_np = (np.abs(b1_t[0].detach().cpu().numpy()) > 0)

    coords = create_meshgrid(matrix_size, device=device)

    if args.ndim == 2:
        n_acs = int(0.04 * matrix_size[-2])
        full_us_mask = cartesian_undersampling(
            matrix_size, n_acs, args.R, vd=0, poisson=int(args.poisson)
        )
    else:
        n_acs = int(0.04 * matrix_size[-2])
        ms2d  = (matrix_size[-2], matrix_size[-1])
        base  = cartesian_undersampling(ms2d, n_acs, args.R, vd=0, poisson=int(args.poisson))
        full_us_mask = np.repeat(base[None, ...], matrix_size[0], axis=0)

    split_us_mask, val_mask = separate_mask(full_us_mask, args.split)
    full_mask_t = torch.tensor(full_us_mask, device=device, dtype=torch.float32)
    us_mask_t  = torch.tensor(split_us_mask, device=device, dtype=torch.float32)
    val_mask_t = torch.tensor(val_mask, device=device, dtype=torch.float32)

    return DataBundle(
        kdata=kdata_t, b1=b1_t,
        matrix_size=matrix_size, fft_axes=fft_axes, ncoil=ncoil,
        image_ref_np=image_ref_np, mask_np=mask_np,
        coords=coords, us_mask=us_mask_t, val_mask=val_mask_t,
        full_mask=full_mask_t,
    )


def train_mlp(
    *,
    h: HParams,
    args: RunArgs,
    bundle: DataBundle,
    inference: int = 0,
    splitting: int = 1,
    flags: Optional[TrainFlags] = None,
) -> Union[float, str]:
    device = bundle.kdata.device
    cfg: Dict[str, Any] = load_config(args.config)
    opt_cfg: Dict[str, Any] = cfg["optimizer"]

    kdata_t      = bundle.kdata
    b1_t         = bundle.b1
    matrix_size  = bundle.matrix_size
    fft_axes     = bundle.fft_axes
    image_ref_np = bundle.image_ref_np
    mask_np      = bundle.mask_np
    coords       = bundle.coords
    us_mask_t    = bundle.us_mask
    full_mask_t  = bundle.full_mask

    # retrospective undersampling
    mask_for_train = full_mask_t if (inference or not splitting) else us_mask_t
    kdata_us = kdata_t * mask_for_train

    if inference:
        with torch.no_grad():
            image_us = coil_combine(kdata_us, b1_t, domain="kspace", mode="sense",
                                    coil_axis=0, fft_ndims=args.ndim)
            image_us_np = np.squeeze(image_us.detach().cpu().numpy())
            nrm0, ssm0, psn0 = img_evaluation(image_us_np, image_ref_np, mask_np)
        if args.ndim == 2:
            plot_image(np.abs(image_us_np),
                       f"{args.outdir}/us_img.png",
                       f"IFFT recon\nNRMSE:{nrm0:.2f} SSIM:{ssm0:.2f} PSNR:{psn0:.2f}",
                       dpi=200, vmax_scale=args.vmax_scale)
            plot_image(np.abs(mask_for_train.detach().cpu().numpy()),
                       f"{args.outdir}/us_pattern.png",
                       f"undersampling pattern R={args.R}")

    model, encoder, decoder = create_model(
        cfg, n_in=args.ndim, per_level_scale=h.per_level_scale, n_out=2
    )
    model = model.to(device)
    enc_params = list(encoder.parameters())
    net_params = list(decoder.parameters())

    # distinct weight decays for encoder (Tikhonov) and MLP
    opt = torch.optim.Adam(
        [
            {"params": enc_params, "weight_decay": h.wd_enc},
            {"params": net_params, "weight_decay": h.wd_mlp},
        ],
        lr=h.lr,
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg["epsilon"],
    )

    dc_history = []

    for step in range(args.n_steps):
        out_xy  = model(coords).reshape(*matrix_size, 2).to(torch.float32)
        out_img = torch.complex(out_xy[..., 0], out_xy[..., 1])

        img_ncoil = out_img.unsqueeze(0) * b1_t
        out_ks    = fftnd_torch(img_ncoil, axes=fft_axes)
        out_ks    = out_ks * mask_for_train

        # Eq. 8 in the manuscript
        pred   = out_ks.unsqueeze(0)
        target = kdata_us.unsqueeze(0)
        loss_dc = self_weighted_l2_loss(pred, target, h.eps)

        opt.zero_grad(set_to_none=True)
        loss_dc.backward()
        opt.step()

        dc_history.append(loss_dc.item())

        if inference and ((step % args.log_every) == 0 or step == args.n_steps - 1):
            os.makedirs(args.outdir, exist_ok=True)
            with torch.no_grad():
                out_np = np.squeeze(out_img.detach().cpu().numpy())
                nrm, ssm, psn = img_evaluation(out_np, image_ref_np, mask_np)
                title = f"INR Recon {step} \n" \
                        f"NRMSE:{nrm:.3f} SSIM:{ssm:.3f} PSNR:{psn:.2f}dB"
                if args.ndim == 2:
                    plot_image(np.abs(out_np),
                            f"{args.outdir}/recon_{step:05d}.png",
                            title, dpi=200, vmax_scale=args.vmax_scale)
                else:
                    nrows, ncols = compute_grid(matrix_size[0])
                    plot_images_multi(np.abs(out_np),
                                    f"{args.outdir}/recon_{step:05d}.png",
                                    title, nrows, ncols, dpi=200)

    with torch.no_grad():
        out_xy  = model(coords).reshape(*matrix_size, 2).to(torch.float32)
        out_img = torch.complex(out_xy[..., 0], out_xy[..., 1])

    if inference:
        with torch.no_grad():
            out_np = np.squeeze(out_img.detach().cpu().numpy())
            nrm, ssm, psn = img_evaluation(out_np, image_ref_np, mask_np)
        return (f"Inference done. \nFinal NRMSE={nrm:.3f} SSIM={ssm:.3f} PSNR={psn:.2f} dB;\n"
                f"hparams={h.to_dict()}")

    # upper-level objective: oracle > args.upper_obj; no-split falls back to DC history
    chosen = "oracle" if (flags and flags.oracle) else args.upper_obj

    if splitting:
        with torch.no_grad():
            if chosen == "oracle":
                out_np = np.squeeze(out_img.detach().cpu().numpy())
                return float(np.linalg.norm(out_np - image_ref_np))

            # k-space validation on held-out mask
            val_ks_pred = fftnd_torch(out_img.unsqueeze(0) * b1_t, axes=fft_axes)
            val_ks_pred = val_ks_pred * bundle.val_mask
            val_ks_ref  = kdata_t * bundle.val_mask

            if chosen == "frobenius":
                diff = val_ks_ref - val_ks_pred
                return float(torch.norm(diff).item())

            val_loss = self_weighted_l2_loss(
                pred=val_ks_pred.unsqueeze(0),
                target=val_ks_ref.unsqueeze(0),
                eps=1e-4,  # fixed; not searched
            ).item()
            return float(val_loss)
    else:
        k = min(100, len(dc_history))
        return float(np.mean(dc_history[-k:])) if k > 0 else float(loss_dc.item())


def make_objective(run_args: RunArgs, bundle: DataBundle,
                   flags: Optional[TrainFlags] = None, splitting: int = 1):
    trial = {"i": 0, "best": float("inf")}
    def _objective(log_lr, log_wd_enc, log_wd_mlp, log_eps, per_level_scale):
        lr     = 10.0 ** log_lr
        wd_enc = 10.0 ** log_wd_enc
        wd_mlp = 10.0 ** log_wd_mlp
        eps    = 10.0 ** log_eps
        h = HParams(lr=lr, wd_enc=wd_enc, wd_mlp=wd_mlp,
                    eps=eps, per_level_scale=per_level_scale)
        val = train_mlp(h=h, args=run_args, bundle=bundle,
                        inference=0, splitting=splitting, flags=flags)
        trial["i"] += 1
        v = float(val)
        is_best = v < trial["best"]
        if is_best:
            trial["best"] = v
        flag = " *" if is_best else "  "
        print(f"[BayesOpt trial {trial['i']:3d}{flag}] obj={v:.5e} best={trial['best']:.5e} "
              f"| lr={lr:.2e} wd_enc={wd_enc:.2e} wd_mlp={wd_mlp:.2e} "
              f"eps={eps:.2e} pls={per_level_scale:.3f}")
        return -v
    return _objective