# inr/utils/utils.py
from __future__ import annotations
import numpy as np
import torch
from typing import Sequence, Tuple, Union, Optional

# -------------------------------------------------------------------------
# Image-quality metrics
# -------------------------------------------------------------------------

def img_evaluation(output_img, ref_img, mask):
    """NRMSE / SSIM / PSNR placeholder."""
    raise NotImplementedError


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def plot_image(img, path, title, dpi=200, tight=0, vmax_scale=1, colorbar=1):
    """Save a single 2-D image."""
    raise NotImplementedError

def plot_image_diff(*args, **kwargs):
    raise NotImplementedError

def compute_grid(n: int) -> Tuple[int, int]:
    """Return nrows, ncols for √N layout."""
    r = int(np.ceil(np.sqrt(n)))
    return r, int(np.ceil(n / r))

def plot_images_multi(*args, **kwargs):
    raise NotImplementedError

def plot_profile_components(*args, **kwargs):
    raise NotImplementedError


# -------------------------------------------------------------------------
# Meshgrid / FFT wrappers
# -------------------------------------------------------------------------

def create_meshgrid(matrix_size, density=1, device=torch.device("cpu")):
    raise NotImplementedError

def fftnd(*args, **kwargs):  raise NotImplementedError
def ifftnd(*args, **kwargs): raise NotImplementedError
def fftnd_torch(*args, **kwargs):  raise NotImplementedError
def ifftnd_torch(*args, **kwargs): raise NotImplementedError


# -------------------------------------------------------------------------
# Coil-combination & normalisation
# -------------------------------------------------------------------------

def coil_comb(*args, **kwargs):        raise NotImplementedError
def coil_comb_torch(*args, **kwargs):  raise NotImplementedError
def rms_comb_torch(*args, **kwargs):   raise NotImplementedError
def coil_comb_img(*args, **kwargs):    raise NotImplementedError
def image_normalization(*args, **kwargs): raise NotImplementedError


# -------------------------------------------------------------------------
# Sampling masks & splitting
# -------------------------------------------------------------------------

def cartesian_undersampling(*args, **kwargs):          raise NotImplementedError
def sample_center_block(*args, **kwargs):              raise NotImplementedError
def cartesian_undersampling_slice(*args, **kwargs):    raise NotImplementedError
def separate_mask(*args, **kwargs):                    raise NotImplementedError


# -------------------------------------------------------------------------
# MBIR
# -------------------------------------------------------------------------

class MBIR:
    def __init__(self, *_, **__):                 raise NotImplementedError
    def POGM_l1wavelet(self, *_, **__):           raise NotImplementedError
    def CGSENSE(self, *_, **__):                  raise NotImplementedError
    def FBPD_L1TV(self, *_, **__):                raise NotImplementedError


# -------------------------------------------------------------------------
# Misc. maths & helpers
# -------------------------------------------------------------------------

def diff_image(*args, **kwargs):                  raise NotImplementedError
def coil_compress_svd(*args, **kwargs):           raise NotImplementedError
def radon_transform(*args, **kwargs):             raise NotImplementedError
def gen_mask(*args, **kwargs):                    raise NotImplementedError
def gen_mask_vd(*args, **kwargs):                 raise NotImplementedError
def noise_prewhittening(*args, **kwargs):         raise NotImplementedError
def img_noise_simulation(*args, **kwargs):        raise NotImplementedError
def zero_padding(*args, **kwargs):                raise NotImplementedError
def slice_reorder(*args, **kwargs):               raise NotImplementedError
def match_dense_idx(*args, **kwargs):             raise NotImplementedError
def data_cropping(*args, **kwargs):               raise NotImplementedError
def checkerboard_perturbation(*args, **kwargs):   raise NotImplementedError
def impulse_perturbation(*args, **kwargs):        raise NotImplementedError


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def get_args():
    """Return argparse.Namespace with dummy defaults"""
    import argparse, datetime
    ns = argparse.Namespace()
    ns.config = "config.json"
    ns.n_steps = 100
    ns.date = datetime.date.today().isoformat()
    ns.R = 4
    return ns