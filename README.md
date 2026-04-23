# bilevel_optimized_INR

Code for *Bilevel Optimized Implicit Neural Representation for Scan-Specific
Accelerated MRI Reconstruction*.

Paper: **https://doi.org/10.1109/TMI.2026.3686724** (IEEE TMI, 2026)
Preprint: https://arxiv.org/abs/2502.21292

## Repo layout

```text
bilevel_optimized_INR/
├── pyproject.toml              # package metadata & deps
├── README.md
├── LICENSE
├── inr/
│   ├── __init__.py
│   ├── config/
│   │   └── config_cartesian.json
│   ├── load_config.py          # JSON / commentJSON config loader
│   ├── train.py                # INR training + BayesOpt objective
│   ├── model.py                # network setup (tiny-cuda-nn encoder + MLP)
│   ├── data.py                 # loaders for k-space & coil maps
│   ├── loss_fn.py              # self-weighted L2 loss
│   └── utils/
│       ├── __init__.py
│       └── utils.py            # FFT / coil-combine / masks / metrics / plots
├── bilevel_inr.py              # Bilevel optimized INR entry script
└── run.sh                      # example shell script
```

## Installation

Requires Python >= 3.9 and a CUDA-capable GPU (for `tiny-cuda-nn`).

```bash
pip install -e .
```

`tiny-cuda-nn` needs to be built against your CUDA toolchain; follow the
upstream install instructions if the PyPI wheel fails:
https://github.com/NVlabs/tiny-cuda-nn

`mirtorch` is used for the MBIR baseline in `inr/utils/utils.py`; install from
source if a PyPI wheel is unavailable:
https://github.com/guanhuaw/MIRTorch

## Data format

`--data-dir` must contain:

- `kdata.npy` — k-space data, shape `(ncoil, ny, nx)` (2D) or `(ncoil, nz, ny, nx)` (3D).
- `b1.npy`   — coil sensitivity maps, same spatial shape as `kdata`.

## Usage

### Bilevel optimization

```bash
python bilevel_inr.py \
    --mode bilevel \
    --data-dir <path-to-your-data> \
    --outdir  <path-to-output>
```

The Bayesian optimizer searches `lr`, `wd_enc`, `wd_mlp`, `eps` in **log10
space** and `per_level_scale` in linear space; bounds are configurable via
`--lr-min/--lr-max`, etc.

### Oracle (upper-level uses fully-sampled reference)

```bash
python bilevel_inr.py --mode bilevel --upper-obj oracle \
    --data-dir <path-to-your-data> --outdir <path-to-output>
```

### Inference only (with fixed hyperparameters)

```bash
python bilevel_inr.py --mode inference \
    --data-dir <path-to-your-data> --outdir <path-to-output> \
    --lr 1e-3 --wd-enc 4e-4 --wd-mlp 1e-7 \
    --eps 3.7e-4 --per-level-scale 1.52
```

## Citation

```bibtex
@ARTICLE{yu2025bilevel-inr
  author={Yu, Hongze and Fessler, Jeffrey A. and Jiang, Yun},
  journal={IEEE Transactions on Medical Imaging},
  title={Bilevel Optimized Implicit Neural Representation for Scan-Specific Accelerated MRI Reconstruction},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2026.3686724}}
```

## License

MIT — see [LICENSE](LICENSE).
