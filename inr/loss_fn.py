import torch

def self_weighted_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Eq. 8; Tikhonov terms are folded into Adam's weight_decay
    self_weighting = torch.stack((torch.absolute(pred), torch.absolute(pred)), dim=-1).detach()

    pred = torch.view_as_real(pred)
    target = torch.view_as_real(target)
    diff = pred - target
    loss = diff**2 / (self_weighting**2 + eps)
    return loss.mean()
