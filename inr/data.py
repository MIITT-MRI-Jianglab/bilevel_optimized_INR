# inr/data.py
import numpy as np
import torch

def load_kspace(folder: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load k-space data and coil maps from `folder`.
    """
    kdata = np.load(f"{folder}/kdata.npy")  # shape: (coil, ... )
    b1    = np.load(f"{folder}/b1.npy")     # shape: (coil, ... )
    return torch.tensor(kdata), torch.tensor(b1)