from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from typing import Sequence, Tuple, Union, Optional, Literal

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse


# -------------------------------------------------------------------------
# Image-quality metrics
# -------------------------------------------------------------------------

def img_evaluation(output_img, ref_img, mask):
    """
    Evaluate NRMSE, SSIM, and PSNR of the output image compared with the reference image.
    Both images are assumed to be magnitude images (normalized to [0,1]).
    
    If the input images are 3D (shape: [nslice, height, width]), the metrics are computed
    slice by slice (using the corresponding 2D mask) and then averaged.
    
    Parameters:
        output_img (ndarray): Reconstructed image (2D or 3D).
        ref_img (ndarray): Reference image (2D or 3D).
        mask (ndarray): Mask (2D or 3D) to apply to the images.
    
    Returns:
        list: [NRMSE, SSIM, PSNR]
    """
    # Compute the absolute value of the images.
    ref_img = np.abs(ref_img)
    output_img = np.abs(output_img)
    
    # Apply the mask if any nonzero elements are present.
    if mask.any():
        ref_img = ref_img * mask
        output_img = output_img * mask

    # If images are 2D.
    if ref_img.ndim == 2:
        # Scale the output to best match the reference.
        scale = np.sum(ref_img * output_img) / np.sum(output_img * output_img)
        output_img_scaled = output_img * scale
        
        # Compute the metrics.
        nrmse_val = nrmse(ref_img, output_img_scaled)
        # Use the full data range from the reference.
        data_range = ref_img.max() - ref_img.min()
        ssim_val = ssim(ref_img, output_img_scaled, data_range=data_range)
        psnr_val = psnr(ref_img, output_img_scaled, data_range=data_range)
        return [nrmse_val, ssim_val, psnr_val]

    # If images are 3D (assumed shape: [nslice, height, width]).
    elif ref_img.ndim == 3:
        nslice = ref_img.shape[0]
        nrmse_list = []
        ssim_list = []
        psnr_list = []
        for i in range(nslice):
            ref_slice = ref_img[i]
            out_slice = output_img[i]
            # If the mask is 3D, select the i-th slice.
            if mask.ndim == 3:
                mask_slice = mask[i]
                ref_slice = ref_slice * mask_slice
                out_slice = out_slice * mask_slice
            # Scale the slice output.
            scale = np.sum(ref_slice * out_slice) / np.sum(out_slice * out_slice)
            out_slice_scaled = out_slice * scale
            # Compute metrics for the slice.
            nrmse_list.append(nrmse(ref_slice, out_slice_scaled))
            data_range = ref_slice.max() - ref_slice.min()
            ssim_list.append(ssim(ref_slice, out_slice_scaled, data_range=data_range))
            psnr_list.append(psnr(ref_slice, out_slice_scaled, data_range=data_range))
        # Return the average metrics over slices.
        return [np.mean(nrmse_list), np.mean(ssim_list), np.mean(psnr_list)]
    
    else:
        raise ValueError("img_evaluation expects 2D or 3D input images.")

# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def diff_image(output_img, ref_img):
    '''
    Calculate the difference image
    '''
    scale = np.sum(output_img * np.conj(ref_img)) / np.sum(output_img * np.conj(output_img))
    output_img *= scale
    return abs(ref_img-output_img)

def plot_image(img, path, title, 
                dpi=200, tight=0, vmax_scale = 1, colorbar = 1):
    '''
        Plot 2D magnitude image/ks and save to the corresponding folder
    '''
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    height, width = img.shape

    # Calculate vmin and vmax for windowing
    vmin = np.min(img)
    vmax = vmax_scale * np.max(abs(img))

    plt.figure()
    plt.imshow(img.T, aspect='auto', cmap='gray', origin='lower', 
                vmin=vmin, vmax=vmax)
    plt.axis('square')
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.title(title)

    if dpi:
        plt.savefig(path, dpi=dpi, transparent=False)
    elif tight:
        plt.savefig(path, bbox_inches='tight', 
                pad_inches=0, transparent=False) 
    else:
        plt.savefig(path, transparent=False)
    plt.close()
    
def compute_grid(n: int) -> Tuple[int, int]:
    """Return nrows, ncols for √N layout."""
    r = int(np.ceil(np.sqrt(n)))
    return r, int(np.ceil(n / r))

def plot_images_multi(arr, path, title, nrows, ncols, dpi=300):
    """
    Plot multiple 2D magnitude images in one figure and save them to a single file.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    arr = np.abs(arr) 
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j < arr.shape[0]:
                axes[i, j].imshow(arr[i * ncols + j], cmap='gray', origin='lower')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    plt.tight_layout()
    fig.suptitle(title)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def plot_profile_components(
    profile_dense: Union[np.ndarray, Sequence],
    profile_orig:  Union[np.ndarray, Sequence],
    matrix_size:   Tuple[int, ...],
    density:       int                 = 1,
    fov:           float               = 1.0,
    colors:        Sequence[str]       = ("tab:blue", "tab:orange"),
    labels:        Sequence[str]       = ("Target", "Original"),
    titles:        Sequence[str]       = ("Magnitude", "Real part", "Imag part"),
    dpi:           int                 = 300,
    save_path:     Optional[str]       = None,
    tight:         bool                = True,
):
    """
    Plot |x|, Re{x}, Im{x} for two 1-D complex profiles that share the same FOV.
    Use this to compare INR output with different resolution (density).

    Parameters
    ----------
    profile_dense : 1-D complex array
        High-density profile sampled on `Nx * density` points.
    profile_orig  : 1-D complex array
        Original profile sampled on `Nx` points.
    matrix_size   : tuple
        Original (non-dense) spatial matrix size, e.g. (Ny, Nx) or (Nz, Ny, Nx).
        Only the *last* element (Nx) is used for the horizontal axis.
    density       : int, default 1
        Spatial density factor used for the dense grid.
    fov           : float, default 1.0
        Physical field-of-view.  Set to 1.0 to keep a normalised x-axis.
    """
    Nx  = matrix_size[-1]
    Nxd = Nx * density

    # --- Sanity checks -----------------------------------------------------
    profile_dense = np.asarray(profile_dense).ravel()
    profile_orig  = np.asarray(profile_orig ).ravel()
    if len(profile_dense) != Nxd:
        raise ValueError(f"len(profile_dense) = {len(profile_dense)} "
                         f"but expected {Nxd} (Nx*density).")
    if len(profile_orig)  != Nx:
        raise ValueError(f"len(profile_orig) = {len(profile_orig)} "
                         f"but expected {Nx}.")

    # --- x‑coordinates (half‑voxel offset, matches create_meshgrid) --------
    hx = 0.5 / Nx
    x_orig = np.linspace(hx, 1 - hx, Nx)
    hx_dense = 0.5 / Nxd
    x_dense = np.linspace(hx_dense, 1 - hx_dense, Nxd)

    # --- force exact overlap of coarse points on the dense axis ----------
    if density % 2 == 1:
        # integer mapping j = i*d + (d-1)/2  <=>  i*d + density//2
        idxs = np.arange(Nx) * density + density // 2
        x_orig_on_dense = x_dense[idxs]
    else:
        # fallback: use the floating one
        x_orig_on_dense = x_orig
    
    # --- Plotting -----------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Row 0 : magnitude ------------------------------------------------------
    axes[0].scatter(x_dense, np.abs(profile_dense),
                    color=colors[0], s=12, alpha=0.7, label=labels[0])
    axes[0].scatter(x_orig_on_dense,  np.abs(profile_orig),
                    color=colors[1], s=12, edgecolor="k", label=labels[1])
    axes[0].set_ylabel("|x|")
    axes[0].set_title(titles[0])
    axes[0].legend(loc="best")

    # Row 1 : real part ------------------------------------------------------
    axes[1].scatter(x_dense, np.real(profile_dense),
                    color=colors[0], s=12, alpha=0.7)
    axes[1].scatter(x_orig_on_dense,  np.real(profile_orig),
                    color=colors[1], s=12, edgecolor="k")
    axes[1].set_ylabel("Re{x}")
    axes[1].set_title(titles[1])

    # Row 2 : imaginary part -------------------------------------------------
    axes[2].scatter(x_dense, np.imag(profile_dense),
                    color=colors[0], s=12, alpha=0.7)
    axes[2].scatter(x_orig_on_dense,  np.imag(profile_orig),
                    color=colors[1], s=12, edgecolor="k")
    axes[2].set_ylabel("Im{x}")
    axes[2].set_xlabel("Sampling Location")
    axes[2].set_title(titles[2])

    if tight:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

def plot_training_history():
    pass

# -------------------------------------------------------------------------
# Meshgrid / FFT wrappers
# -------------------------------------------------------------------------

def create_meshgrid(matrix_size, density=1, device=torch.device("cuda")):
    """
    Create a (N, D) grid of normalized coordinates in [0,1] for an
    implicit network, where D = len(matrix_size) is 2 or 3.

    Args:
        matrix_size: tuple of integers, the size of the image and k-space.
                     For 2D, it should be (height, width).
                     For 3D, it should be (depth, height, width).
        density: float, the density of the grid points.
                Default is 1.0 (full resolution).
        device: torch.device, the device to create the tensor on.

    Returns:
        coords: torch.Tensor of shape (prod(matrix_size), D)
                and dtype torch.float32 on `device`.
    """
    D = len(matrix_size)
    # Multiply the matrix size by the density factor
    matrix_size = [int(size * density) for size in matrix_size]
    if D == 3:
        Z, Y, X = matrix_size           # depth, height, width
        # half‑voxel offsets
        hz = 0.5 / Z 
        hy = 0.5 / Y
        hx = 0.5 / X
        # 1D linspaces
        zs = torch.linspace(hz, 1 - hz, Z, device=device)
        ys = torch.linspace(hy, 1 - hy, Y, device=device)
        xs = torch.linspace(hx, 1 - hx, X, device=device)
        zv, yv, xv = torch.meshgrid(zs, ys, xs, indexing="ij")
        coords = torch.stack([zv, yv, xv], dim=-1)
        return coords.reshape(-1, 3)
    elif D == 2:
        Y, X = matrix_size               # height, width
        hy = 0.5 / Y
        hx = 0.5 / X
        ys = torch.linspace(hy, 1 - hy, Y, device=device)
        xs = torch.linspace(hx, 1 - hx, X, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yv, xv], dim=-1)
        return coords.reshape(-1, 2)
    else:
        raise ValueError(f"Only 2D or 3D matrix_size supported, got {D}-D.")

def _to_axes_tuple(x: Optional[Sequence[int]], ndim: int) -> tuple[int, ...]:
    if x is None:
        return tuple(range(ndim))
    return tuple(x)

def fftnd(img, axes: Optional[Sequence[int]] = (1, -1)):
    """
    NumPy N-D FFT with unitary-like normalization on `axes`.
    """
    axes_t = _to_axes_tuple(axes, img.ndim)
    img = np.fft.ifftshift(img, axes=axes_t)
    kspace = np.fft.fftn(img, axes=axes_t)
    kspace = np.fft.fftshift(kspace, axes=axes_t)
    sizes = [img.shape[ax] for ax in axes_t]
    kspace /= np.sqrt(np.prod(sizes))
    return kspace

def ifftnd(kspace, axes: Optional[Sequence[int]] = (1, -1)):
    """
    NumPy N-D IFFT with inverse normalization on `axes`.
    """
    axes_t = _to_axes_tuple(axes, kspace.ndim)
    kspace = np.fft.ifftshift(kspace, axes=axes_t)
    img = np.fft.ifftn(kspace, axes=axes_t)
    img = np.fft.fftshift(img, axes=axes_t)
    sizes = [kspace.shape[ax] for ax in axes_t]
    img *= np.sqrt(np.prod(sizes))
    return img

def ifftnd_torch(kspace, axes: Optional[Sequence[int]] = (-2, -1)):
    """
    PyTorch N-D IFFT with inverse normalization on `axes`.
    """
    axes_t = _to_axes_tuple(axes, kspace.ndim)
    kspace = torch.fft.ifftshift(kspace, dim=axes_t)
    img = torch.fft.ifftn(kspace, dim=axes_t)
    img = torch.fft.fftshift(img, dim=axes_t)
    sizes = torch.tensor([img.shape[ax] for ax in axes_t],
                         device=img.device, dtype=img.real.dtype if img.is_complex() else img.dtype)
    img *= torch.sqrt(sizes.prod())
    return img

def fftnd_torch(img, axes: Optional[Sequence[int]] = (-2, -1)):
    """
    PyTorch N-D FFT with unitary-like normalization on `axes`.
    """
    axes_t = _to_axes_tuple(axes, img.ndim)
    img = torch.fft.ifftshift(img, dim=axes_t)
    kspace = torch.fft.fftn(img, dim=axes_t)
    kspace = torch.fft.fftshift(kspace, dim=axes_t)
    sizes = torch.tensor([img.shape[ax] for ax in axes_t],
                         device=img.device, dtype=img.real.dtype if img.is_complex() else img.dtype)
    kspace /= torch.sqrt(sizes.prod())
    return kspace
    
# -------------------------------------------------------------------------
# Coil-combination & normalisation
# -------------------------------------------------------------------------

Array = Union[np.ndarray, torch.Tensor]
def coil_combine(
    data: Array,
    cmap: Array,
    *,
    domain: Literal["kspace", "image"] = "kspace",
    mode:   Literal["sense", "rss"]    = "sense",
    coil_axis: int = 0,
    fft_ndims: int = 2,
    keepdims: bool = False,
) -> Array:
    """
    Unified coil combination for NumPy and PyTorch.

    Args:
        data:  k-space or image data with a coil dimension (see `coil_axis`).
        cmap:  coil sensitivity maps (image-domain) broadcastable to `data`
               after IFFT if domain=="kspace".
        domain: "kspace" -> IFFT over last `fft_ndims` axes first; "image" -> use as-is.
        mode:   "sense" -> sum(img * conj(cmap)) over coils;
                "rss"   -> sqrt(sum(|img|^2)) over coils.
        coil_axis: index of the coil dimension in `data` (e.g., 0).
        fft_ndims: 2 for 2D, 3 for 3D (axes are the last `fft_ndims` dims).
        keepdims: keep the coil axis after reduction.

    Returns:
        Combined image (complex for "sense"; real magnitude for "rss").
    """
    is_torch = torch.is_tensor(data)
    xp = torch if is_torch else np

    # Helpers that normalize numpy/torch API differences
    def _sum(x, axis, keepdims):
        if is_torch:
            return torch.sum(x, dim=axis, keepdim=keepdims)
        else:
            return np.sum(x, axis=axis, keepdims=keepdims)

    def _abs(x):  return torch.abs(x) if is_torch else np.abs(x)
    def _conj(x): return torch.conj(x) if is_torch else np.conj(x)
    def _sqrt(x): return torch.sqrt(x) if is_torch else np.sqrt(x)

    cmap = xp.squeeze(cmap)
    if domain == "kspace":
        axes = tuple(range(-fft_ndims, 0))  # last fft_ndims axes
        img = ifftnd_torch(data, axes) if is_torch else ifftnd(data, axes)
    elif domain == "image":
        img = data
    else:
        raise ValueError("domain must be 'kspace' or 'image'")

    if mode == "sense":
        # SENSE-like: sum_c img_c * conj(s_c)
        out = _sum(img * _conj(cmap), axis=coil_axis, keepdims=keepdims)
    elif mode == "rss":
        # Root-sum-of-squares magnitude across coils
        out = _sqrt(_sum(_abs(img) ** 2, axis=coil_axis, keepdims=keepdims))
    else:
        raise ValueError("mode must be 'sense' or 'rss'")

    return out

# -------------------------------------------------------------------------
# Sampling masks & splitting
# -------------------------------------------------------------------------

def cartesian_undersampling(matrix_size, n_acs, R, vd=0, skipping_four=0, poisson=0, offset=0):
    '''
    Simulate 2D cartesian undersampling mask

    Parameters:
    - matrix_size: the size of the image and k-space
    - n_acs: Leave how many center k-space line for auto-calibration
    - R: overall how many acceleration ratio
    - vd: whether to use variable density sampling

    Returns:
    - mask: a binary mask indicating sampled (1) and unsampled (0) locations
    '''
    n_pe, n_fe = matrix_size # total number of phase encoding lines
    mask = np.zeros(matrix_size)
    if not poisson:
        start_acs = n_pe // 2 - n_acs // 2
        end_acs = start_acs + n_acs
        mask[start_acs:end_acs, :] = 1

        if vd:
            gaussian = np.exp(-np.square(np.arange(n_pe) - n_pe / 2) / (2 * (n_pe / 6)**2))
            gaussian /= gaussian.sum()  # Normalize to sum to 1
            num_samples = int(n_pe / R)
            samples = np.random.choice(n_pe, size=num_samples, replace=False, p=gaussian)
            for s in samples:
                mask[s, :] = 1
        else:
            if R > 4 and skipping_four:
                mask[::4,:] = 1
                nline_to_discard = (n_pe // 4 - n_pe // R)
                sampled_lines = np.where(mask[:,0] == 1)[0] # Get indices of sampled lines
                outer_lines = np.setdiff1d(sampled_lines, np.arange(start_acs, end_acs)) # Exclude ACS region
                
                # Discard lines from the outer regions
                discard_from_each_side = nline_to_discard // 2
                mask[outer_lines[:discard_from_each_side], :] = 0 # Discard from the beginning
                mask[outer_lines[-discard_from_each_side:], :] = 0 # Discard from the end
            else:
                mask[offset::R, :] = 1
    else:
        from sigpy import mri
        mask = mri.poisson(matrix_size, R, (0,0), crop_corner=False, seed=7)
        # Add center n_acs x n_acs block
        start_pe = n_pe // 2 - n_acs // 2
        end_pe = start_pe + n_acs
        start_fe = n_fe // 2 - n_acs // 2
        end_fe = start_fe + n_acs
        mask[start_pe:end_pe, start_fe:end_fe] = 1
    # return mask.astype(bool)
    return mask

def sample_center_block(
    matrix_size: Tuple[int, int],
    block_size: Union[int, Tuple[int, int]],
    as_bool: bool = False
) -> np.ndarray:
    """
    Generate a Cartesian undersampling mask that only samples
    a centered block (square or rectangular) in k-space.

    Args:
        matrix_size:  (n_pe, n_fe) total number of phase- and freq-encoding lines
        block_size:   either
                         - int: side-length of a square block, or
                         - (block_pe, block_fe): block height & width
        as_bool:      if True, return mask.astype(bool) instead of int

    Returns:
        mask: (n_pe, n_fe) array with 1’s in the central block, 0’s elsewhere
    """
    n_pe, n_fe = matrix_size

    # unpack block_size
    if isinstance(block_size, tuple):
        block_pe, block_fe = block_size
    else:
        block_pe = block_fe = block_size

    # compute center indices
    start_pe = n_pe // 2 - block_pe // 2
    end_pe   = start_pe + block_pe
    start_fe = n_fe // 2 - block_fe // 2
    end_fe   = start_fe + block_fe

    # build mask
    mask = np.zeros((n_pe, n_fe), dtype=int)
    mask[start_pe:end_pe, start_fe:end_fe] = 1

    return mask.astype(bool) if as_bool else mask

def cartesian_undersampling_slice(matrix_size, n_acs, R, vd=0):
    '''
    Simulate 3D cartesian undersampling mask, undersampling only the slice dimension (nslice)
    while keeping ny and nx fully sampled.

    Parameters:
    - matrix_size: a tuple (nslice, ny, nx), the size of the image and k-space
    - n_acs: Leave how many center k-space slices for auto-calibration
    - R: overall acceleration ratio along the slice dimension (nslice)
    - vd: whether to use variable density sampling for the slice dimension

    Returns:
    - mask: a binary mask indicating sampled (1) and unsampled (0) locations with shape (nslice, ny, nx)
    '''
    nslice, ny, nx = matrix_size # slice, y, x dimensions
    mask = np.zeros((nslice, ny, nx))
    
    # Create ACS region in the center of the slice dimension
    start_acs = nslice // 2 - n_acs // 2
    end_acs = start_acs + n_acs
    mask[start_acs:end_acs, :, :] = 1

    if vd:
        # Variable density sampling along the slice dimension (nslice)
        gaussian = np.exp(-np.square(np.arange(nslice) - nslice / 2) / (2 * (nslice / 6)**2))
        gaussian /= gaussian.sum()  # Normalize to sum to 1
        num_samples = int(nslice / R)
        samples = np.random.choice(nslice, size=num_samples, replace=False, p=gaussian)
        for s in samples:
            mask[s, :, :] = 1
    else:
        mask[::R, :, :] = 1
    return mask

def separate_mask(mask, proportion, seed=7):
    '''
    Randomly separates the input mask into two masks with a specified proportion.

    Parameters:
    - mask: np.ndarray, binary mask indicating sampled (1) and unsampled (0) locations
    - proportion: float, the proportion of ones in the first mask (0 < proportion < 1)

    Returns:
    - mask1: np.ndarray, first separated mask with specified proportion of ones
    - mask2: np.ndarray, second separated mask with the remaining ones
    '''
    if not (0 < proportion < 1):
        raise ValueError("Proportion must be between 0 and 1")

    # 1) Find all the '1' positions
    ones_idx = np.argwhere(mask == 1)

    # 2) Shuffle
    np.random.seed(seed)
    np.random.shuffle(ones_idx)

    # 3) Split
    cut = int(len(ones_idx) * proportion)
    idx1, idx2 = ones_idx[:cut], ones_idx[cut:]

    # 4) Create empty masks
    mask1 = np.zeros_like(mask)
    mask2 = np.zeros_like(mask)

    # 5) Use tuple-indexing for arbitrary dims
    mask1[tuple(idx1.T)] = 1
    mask2[tuple(idx2.T)] = 1

    return mask1, mask2

# -------------------------------------------------------------------------
# MBIR
# -------------------------------------------------------------------------

class MBIR:
    '''
    Model-based Iterative Reconstruction for MRI
    Based on MIRTorch by Guanhua https://github.com/guanhuaw/MIRTorch

    Input (torch tensors):
    kdata: undersampled k-space data to be reconstructed, shape (nbatch, ncoil, npe, nro) 
    smap: coil maps, shape (nbatch, ncoil, ny, nx)
    us_mask: undersampling mask, shape (nbatch, ncoil, npe, nro)
    '''
    def __init__(self, kdata, smap, us_mask, device='cuda'):
        from mirtorch.linear import FFTCn, Sense
        self.device = torch.device(device)
        self.kdata = kdata.to(self.device).to(torch.complex64)
        self.smap = smap.to(self.kdata)
        self.us_mask = us_mask.to(self.kdata)
        self.kdata_us = self.kdata * self.us_mask
        (_, nc, ny, nx) = smap.shape

        self.Fop = FFTCn((1, nc, nx, ny), (1, nc, nx, ny), (2,3), norm='ortho')
        self.Sop = Sense(self.smap, self.us_mask[0])
        self.I0 = self.Sop.H * self.kdata

    def gradA(self, x):
        return self.Sop.H*self.Sop*x-self.I0

    def lipschitz(self):
        from mirtorch.alg import power_iter
        L = power_iter(self.Sop, torch.randn_like(self.I0), max_iter=200)
        return L

    def POGM_l1wavelet(self, alpha=1e-7):
        from mirtorch.alg import POGM
        from mirtorch.linear import Wavelet2D
        from mirtorch.prox import L1Regularizer
        W = Wavelet2D(self.I0.shape, padding='periodization', J=2, wave_type='db4', device=self.device)
        Prox = L1Regularizer(alpha, T=W)
        evaluation_wavelet = lambda x: (torch.norm(self.Sop*x-self.kdata_us)**2).item()+alpha*torch.norm(Prox(x,1), p=1).item()
        lipschitz_constant = self.lipschitz()[1].item()

        [pg_wavelet, loss_pg_wavelet] = POGM(f_grad=self.gradA, f_L=lipschitz_constant**2, g_prox=Prox, max_iter=100, eval_func=evaluation_wavelet).run(x0=self.I0)
        pg_wavelet_np = pg_wavelet[0,0,:,:].cpu().data.numpy()
        return pg_wavelet_np

    def CGSENSE(self, alpha=0.001):
        from mirtorch.alg.cg import CG
        from mirtorch.linear import Diff2dgram
        T = Diff2dgram(self.Sop.size_in)
        CG_tik = CG(self.Sop.H*self.Sop+alpha*T, max_iter=100, tol=1e-12, alert=False)
        I_tik = CG_tik.run(self.I0, self.I0)
        return np.squeeze(I_tik.cpu().numpy())

    def FBPD_L1TV(self, alpha=1e-6):
        from mirtorch.alg import FBPD
        from mirtorch.linear import Diffnd
        from mirtorch.prox import L1Regularizer, Const
        T = Diffnd(self.I0.shape, [2,3])
        P1 = L1Regularizer(alpha)
        evaluation_l1tv = lambda x: (torch.norm(self.Sop*x-self.kdata_us)**2).item()+alpha*torch.norm(T*x, p=1).item()
        f_prox = Const()
        lipschitz_constant = self.lipschitz()[1].item()

        [fbpd_tv, loss_fbpd_tv] = FBPD(self.gradA, f_prox, P1, lipschitz_constant**2, 8, G=T, max_iter=100, eval_func=evaluation_l1tv).run(self.I0)
        fbpd_tv_np = fbpd_tv[0,0,:,:].cpu().data.numpy()
        return fbpd_tv_np

# -------------------------------------------------------------------------
# Misc. helpers
# -------------------------------------------------------------------------

def match_dense_idx(coarse_idx, matrix_size, density):
    """
    Find the index on the dense grid whose physical coordinate
    best matches the coarse grid row at `coarse_idx`.

    Args:
        coarse_idx  (int): row index in the coarse image (0 ≤ coarse_idx < M)
        matrix_size (tuple[int,int]): (height, width) of the coarse image
        density     (int): up-sampling factor along each axis

    Returns:
        int: the row index in the dense image (0 ≤ idx < M*density)
    """
    M = matrix_size[0]
    d = density

    # normalized positions of row centers
    ys_coarse = np.linspace(0.5/M,     1 - 0.5/M,     M)
    ys_dense  = np.linspace(0.5/(M*d), 1 - 0.5/(M*d), M*d)

    y_target = ys_coarse[coarse_idx]
    # pick the dense index whose y is closest
    idx_dense = int(np.argmin(np.abs(ys_dense - y_target)))
    return idx_dense

def data_cropping(kdata, b1):
    # Simple function for removing 2x oversampling in ro dim for fastMRI data
    [nc, nro, npe] = kdata.shape
    if nro % 2 == 0:
        b1 = b1[:, nro//4:-nro//4, :]
        tmp_img = ifftnd(kdata, [-2,-1])
        tmp_img = tmp_img[:, nro//4:-nro//4, :]
        kdata = fftnd(tmp_img, [-2,-1])
        return kdata, b1
    else:
        raise ValueError("nro must be even for fastMRI data cropping")

def impulse_perturbation(r_pos: int,
                         c_pos: int,
                         mag: float,
                         img_size: tuple[int, int]) -> np.ndarray:
    """
    Generate a 2-D impulse (delta) perturbation.

    Parameters
    ----------
    r_pos : int
        Row index (0-based) of the impulse position.
    c_pos : int
        Column index (0-based) of the impulse position.
    mag   : float | complex
        Magnitude (and phase, if complex) of the impulse.
    img_size : (int, int)
        Height-and-width of the output array.

    Returns
    -------
    perturb : np.ndarray
        Array of shape `img_size` with all zeros except one entry = `mag`.
    """
    perturb = np.zeros(img_size, dtype=np.asarray(mag).dtype)  # dtype matches `mag`
    perturb[r_pos, c_pos] = mag
    return perturb
