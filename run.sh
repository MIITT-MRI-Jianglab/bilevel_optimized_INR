#!/bin/bash
set -eu

DATA_DIR="<path-to-your-data>"   # folder containing kdata.npy and b1.npy
OUT_DIR="<path-to-output>"       # where recon images + logs are written
mkdir -p "${OUT_DIR}"

# Inference with fixed hparams (no search). Requires all 5 hparams.
# python bilevel_inr.py \
#   --mode inference \
#   --data-dir "${DATA_DIR}" \
#   --outdir   "${OUT_DIR}" \
#   --lr 1.01e-3 --wd-enc 4.02e-4 --wd-mlp 1e-7 \
#   --eps 3.73e-4 --per-level-scale 1.524

# Bilevel optimization, self-weighted upper-level loss (self-supervised, default).
python bilevel_inr.py \
  --mode bilevel \
  --data-dir "${DATA_DIR}" \
  --outdir   "${OUT_DIR}"

# Oracle bilevel (needs fully-sampled reference; upper-level = image-domain L2).
# python bilevel_inr.py \
#   --mode bilevel \
#   --data-dir "${DATA_DIR}" \
#   --outdir   "${OUT_DIR}" \
#   --upper-obj oracle
