# inr/model.py
import copy, math, torch, tinycudann as tcnn

def create_model(cfg: dict, n_in: int, per_level_scale: float):
    """Return (model, encoder, decoder) - weights are not initialized."""
    enc_cfg = copy.deepcopy(cfg["encoding"])
    enc_cfg["per_level_scale"] = per_level_scale
    enc  = tcnn.Encoding(n_in, enc_cfg)
    dec  = tcnn.Network(enc.n_output_dims, 2, cfg["network"])
    return torch.nn.Sequential(enc, dec), enc, dec

def build_offset_table(enc):
    """Mimic the hash-grid offset logic."""
    L = enc.encoding_config["n_levels"]
    return torch.arange(L + 1, dtype=torch.int32)