import torch


def self_weighted_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Eq. 8; Tikhonov terms are folded into Adam's weight_decay.
    # Note: this averages over ALL elements. If `pred` and `target` have
    # been zero-masked at unacquired k-space locations, those zero pixels
    # still count in the denominator of mean() — diluting the loss by
    # the undersampling rate. Use `self_weighted_l2_loss_masked` instead
    # to average only over acquired locations.
    self_weighting = torch.stack((torch.absolute(pred), torch.absolute(pred)), dim=-1).detach()

    pred = torch.view_as_real(pred)
    target = torch.view_as_real(target)
    diff = pred - target
    loss = diff**2 / (self_weighting**2 + eps)
    return loss.mean()


def self_weighted_l2_loss_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Self-weighted L2 averaged ONLY over acquired k-space locations.

    Same numerator as `self_weighted_l2_loss`, but the mean is taken over
    the count of acquired entries (`mask == 1`) instead of the full
    tensor. This makes the loss magnitude independent of the
    undersampling rate, so hyperparameters tuned at one R value
    transfer cleanly to another.

    Args:
        pred   : complex tensor
        target : complex tensor (same shape as pred; typically already
                 zero-masked at unacquired locations)
        mask   : real/bool tensor broadcastable to pred's complex shape;
                 1 at acquired k-space locations, 0 elsewhere.
        eps    : self-weighting denominator floor.

    Returns:
        Scalar loss = sum(per-element weighted sq) / count(acquired real
        scalars). The "real scalars" count is 2 * (acquired complex
        samples), since we view_as_real on the diff.
    """
    self_weighting = torch.stack(
        (torch.absolute(pred), torch.absolute(pred)), dim=-1,
    ).detach()
    pred_r = torch.view_as_real(pred)
    target_r = torch.view_as_real(target)
    diff = pred_r - target_r
    per_elem = diff ** 2 / (self_weighting ** 2 + eps)

    # Broadcast mask to (..., 2) so each acquired complex sample
    # contributes 2 to the denominator (one for real, one for imag).
    m = mask.to(per_elem.dtype).unsqueeze(-1).expand_as(per_elem)
    num = (per_elem * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den
