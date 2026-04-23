import numpy as np
import torch
from typing import Tuple

def load_kspace(folder: str) -> Tuple[torch.Tensor, torch.Tensor]:
    kdata = np.load(f"{folder}/kdata.npy")
    b1    = np.load(f"{folder}/b1.npy")
    return torch.tensor(kdata), torch.tensor(b1)


def coil_compress_svd(kspace, threshold, select_num=0):
    """SVD-based coil compression for k-space of shape (ncoil, ...)."""
    orig_shape = kspace.shape
    ncoil = orig_shape[0]
    other_shape = orig_shape[1:]
    flat_dim = np.prod(other_shape)

    kspace_2d = kspace.reshape(ncoil, flat_dim)
    U, S, VH = np.linalg.svd(kspace_2d, full_matrices=False)

    total_energy = np.sum(S)
    cumulative_energy = 0.0
    virtual_coils = ncoil
    for i, s_val in enumerate(S):
        cumulative_energy += s_val
        if cumulative_energy >= threshold * total_energy:
            virtual_coils = i + 1
            break

    if select_num:
        virtual_coils = select_num

    ks_compressed_flat = U[:, :virtual_coils].conj().T @ kspace_2d
    ks_compressed = ks_compressed_flat.reshape((virtual_coils,) + other_shape)

    return ks_compressed, U[:, :virtual_coils], S