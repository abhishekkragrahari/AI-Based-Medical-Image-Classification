"""CAM-faithfulness regularisation (CFR), experiment E9 (see EXPERIMENT_PLAN.md).

Two auxiliary training losses that ask the model's own class activation map (CAM) to
behave like a faithful explanation:

* deletion loss: when the top-``q`` fraction of pixels of the true-class CAM is removed,
  the true-class probability should fall to chance (0.5) or below
  (``relu(p_y(x_deleted) - 0.5)``). Related to the attention-mining loss of GAIN
  (Li et al., CVPR 2018).
* equivariance loss: the CAM of an augmented image should equal the augmented CAM of the
  original image (L1 on max-normalised maps). Related to SEAM (Wang et al., CVPR 2020).

Removed pixels are filled with a blurred copy of the image during training, while the
deletion metric in ``medxai.xai`` fills with 0 (the training mean). The model therefore
cannot lower the evaluation metric just by learning to react to the training fill.

Whether this combination is new for chest X-ray classification is a hypothesis to check
against the literature (RESEARCH_GAP.md), not a claim.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


@contextmanager
def frozen_bn_stats(model: nn.Module):
    """Run extra forward passes without letting them change BatchNorm running statistics.

    The extra passes update throw-away copies of the buffers; the originals are put back
    untouched (restoring them in place would break autograd for the main forward pass).
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    names = ("running_mean", "running_var", "num_batches_tracked")
    saved = [{n: getattr(m, n) for n in names} for m in bns]
    for m, s in zip(bns, saved):
        for n, t in s.items():
            setattr(m, n, t.clone())
    try:
        yield
    finally:
        for m, s in zip(bns, saved):
            for n, t in s.items():
                setattr(m, n, t)


def _up(cam: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False)


def _max_norm(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = F.relu(m)
    return m / (m.amax(dim=(-2, -1), keepdim=True) + eps)


def top_fraction_mask(heat: torch.Tensor, q: float) -> torch.Tensor:
    """1 for the top-``q`` fraction of pixels of each map (B x 1 x H x W), else 0."""
    flat = heat.flatten(1)
    k = max(1, int(round(q * flat.shape[1])))
    idx = flat.topk(k, dim=1).indices
    return torch.zeros_like(flat).scatter_(1, idx, 1.0).view_as(heat)


def blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def deletion_loss(model: nn.Module, x: torch.Tensor, feats: torch.Tensor, y: torch.Tensor,
                  q: float = 0.2) -> torch.Tensor:
    with torch.no_grad():
        mask = top_fraction_mask(_up(model.cam(feats.detach(), y), x.shape[-2:]), q)
    x_del = x * (1 - mask) + blur(x) * mask
    with frozen_bn_stats(model):
        p = torch.softmax(model(x_del), 1).gather(1, y[:, None])[:, 0]
    return F.relu(p - 0.5).mean()


def equivariance_loss(model: nn.Module, x_clean: torch.Tensor, theta: torch.Tensor,
                      feats_aug: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    size = x_clean.shape[-2:]
    with frozen_bn_stats(model):
        cam_clean = _up(model.cam(model.features(x_clean), y), size)
    grid = F.affine_grid(theta, list(cam_clean.shape), align_corners=False)
    warped = F.grid_sample(_max_norm(cam_clean), grid, align_corners=False, padding_mode="zeros")
    valid = F.grid_sample(torch.ones_like(cam_clean), grid, align_corners=False, padding_mode="zeros")
    diff = (warped - _max_norm(_up(model.cam(feats_aug, y), size))).abs() * valid
    return diff.sum() / valid.sum().clamp_min(1.0)


def cfr_penalty(model: nn.Module, x_aug: torch.Tensor, feats_aug: torch.Tensor, y: torch.Tensor,
                x_clean: torch.Tensor | None, theta: torch.Tensor | None, del_weight: float,
                eq_weight: float, del_frac: float) -> tuple[torch.Tensor, dict]:
    """Weighted sum of the two losses (either weight may be 0) and their values for logging."""
    total, parts = x_aug.new_zeros(()), {}
    if del_weight:
        d = deletion_loss(model, x_aug, feats_aug, y, del_frac)
        total, parts["del"] = total + del_weight * d, d.item()
    if eq_weight and theta is not None:
        e = equivariance_loss(model, x_clean, theta, feats_aug, y)
        total, parts["eq"] = total + eq_weight * e, e.item()
    return total, parts
