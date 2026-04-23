import copy, math, torch
import tinycudann as tcnn
import torch.nn as nn
from typing import Tuple, Dict, Any

def create_model(cfg: Dict[str, Any],
                 n_in: int,
                 per_level_scale: float,
                 n_out: int = 2) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Return (model, encoder, decoder); overrides encoding per_level_scale."""
    enc_cfg = copy.deepcopy(cfg["encoding"])
    enc_cfg["per_level_scale"] = per_level_scale

    enc = tcnn.Encoding(n_in, enc_cfg)
    dec = tcnn.Network(enc.n_output_dims, n_out, cfg["network"])
    return nn.Sequential(enc, dec), enc, dec


def build_offset_table(enc):
    cfg  = enc.encoding_config
    L    = cfg["n_levels"]
    R0   = cfg["base_resolution"]
    s    = cfg["per_level_scale"]
    Hmax = 1 << cfg["log2_hashmap_size"]
    D    = enc.n_input_dims

    offsets = [0]
    acc = 0
    for l in range(L):
        Rl = math.ceil(R0 * (s ** l))
        n_grid = min(Hmax, Rl ** D)
        n_grid = math.ceil(n_grid / 8) * 8  # tcnn aligns to 8
        acc += n_grid
        offsets.append(acc)

    return torch.tensor(offsets, dtype=torch.int32)