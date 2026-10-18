"""Explainability methods and explanation-quality checks (see XAI_PLAN.md).

Methods: Grad-CAM, Integrated Gradients, occlusion sensitivity.
Checks:  border-mass (shortcut reliance), cross-method/seed rank agreement,
         perturbation stability, deletion-curve faithfulness.

Heatmaps show which pixels influence the model's output; they do not show that
the model reasons in a clinically valid way.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def grad_cam(model: nn.Module, x: torch.Tensor, target: int | None = None,
             layer: nn.Module | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., ICCV 2017) for a single image x of shape 1 x C x H x W.

    Returns an H x W map in [0, 1] upsampled bilinearly to the input size.
    """
    layer = layer if layer is not None else model.gradcam_layer
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        t = int(logits.argmax(1)) if target is None else int(target)
        model.zero_grad()
        logits[0, t].backward()
        a, g = acts["a"].detach(), grads["g"].detach()
    finally:
        h1.remove()
        h2.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _normalise(cam.numpy())


def integrated_gradients(model: nn.Module, x: torch.Tensor, target: int | None = None,
                         baseline: torch.Tensor | None = None, steps: int = 32) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al., ICML 2017), Riemann midpoint rule.

    Baseline defaults to the all-zeros tensor in *normalised* space, i.e. the
    training-set mean intensity. Returns |attribution| normalised to [0, 1].
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    with torch.no_grad():
        t = int(model(x).argmax(1)) if target is None else int(target)
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps
    path = baseline + alphas.view(-1, 1, 1, 1) * (x - baseline)
    path.requires_grad_(True)
    out = model(path)[:, t].sum()
    (g,) = torch.autograd.grad(out, path)
    attr = ((x - baseline) * g.mean(0, keepdim=True))[0].sum(0)
    return _normalise(attr.abs().detach().numpy())


@torch.no_grad()
def occlusion(model: nn.Module, x: torch.Tensor, target: int | None = None, patch: int = 4,
              stride: int = 2, fill: float = 0.0) -> np.ndarray:
    """Occlusion sensitivity (Zeiler & Fergus, ECCV 2014): drop in target probability
    when a patch is replaced by ``fill`` (0 = training mean in normalised space)."""
    model.eval()
    p0 = torch.softmax(model(x), 1)[0]
    t = int(p0.argmax()) if target is None else int(target)
    H, W = x.shape[-2:]
    heat = np.zeros((H, W))
    cnt = np.zeros((H, W))
    batch, coords = [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xo = x.clone()
            xo[..., i : i + patch, j : j + patch] = fill
            batch.append(xo)
            coords.append((i, j))
    probs = torch.cat([torch.softmax(model(torch.cat(batch[k : k + 256])), 1)[:, t]
                       for k in range(0, len(batch), 256)]).numpy()
    for (i, j), p in zip(coords, probs):
        heat[i : i + patch, j : j + patch] += float(p0[t]) - p
        cnt[i : i + patch, j : j + patch] += 1
    heat = heat / np.maximum(cnt, 1)
    return _normalise(np.clip(heat, 0, None))


METHODS = {"gradcam": grad_cam, "ig": integrated_gradients, "occlusion": occlusion}


def explain(model, x, method: str, target=None, **kw) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown XAI method '{method}'; choose from {sorted(METHODS)}")
    return METHODS[method](model, x, target=target, **kw)


# --------------------------------------------------------------------------- #
# Explanation-quality checks
# --------------------------------------------------------------------------- #
def border_mass(heat: np.ndarray, frac: float = 0.1) -> float:
    """Share of total attribution within the outer ``frac`` border of the image.

    Uniform maps give ~1 - (1 - 2*frac)^2 (0.36 for frac=0.1; 0.383 at 28 px because the
    border is rounded to 3 px); much higher values
    suggest reliance on borders / burned-in markers (a shortcut warning sign).
    """
    h = np.asarray(heat, dtype=float)
    H, W = h.shape
    bh, bw = max(1, int(round(H * frac))), max(1, int(round(W * frac)))
    inner = h[bh : H - bh, bw : W - bw].sum()
    total = h.sum()
    return float((total - inner) / total) if total > 0 else float("nan")


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two attribution maps (NaN if either is constant)."""
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def topk_iou(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> float:
    """IoU of the top-k fraction of pixels in two maps."""
    a, b = np.ravel(a), np.ravel(b)
    n = max(1, int(round(k * a.size)))
    sa, sb = set(np.argsort(-a)[:n]), set(np.argsort(-b)[:n])
    return len(sa & sb) / len(sa | sb)


def perturbation_stability(model, x, method: str, sigma: float = 0.05, n: int = 5, seed: int = 0,
                           **kw) -> float:
    """Mean Spearman agreement between the map for x and maps for x + N(0, sigma^2) noise."""
    g = torch.Generator().manual_seed(seed)
    ref = explain(model, x, method, **kw)
    t = int(model(x).argmax(1))
    vals = [rank_agreement(ref, explain(model, x + sigma * torch.randn(x.shape, generator=g), method,
                                        target=t, **kw)) for _ in range(n)]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


@torch.no_grad()
def deletion_auc(model, x, heat: np.ndarray, target: int | None = None, steps: int = 20,
                 fill: float = 0.0) -> float:
    """Area under the target-probability curve as the most-attributed pixels are
    progressively replaced by ``fill`` (Petsiuk et al., BMVC 2018). Lower = more faithful.
    Compare against :func:`random_deletion_auc` for the same image."""
    model.eval()
    t = int(model(x).argmax(1)) if target is None else int(target)
    order = np.argsort(-np.ravel(heat))
    n = order.size
    xs = []
    for s in range(steps + 1):
        xd = x.clone().view(1, x.shape[1], -1)
        k = int(round(n * s / steps))
        if k:
            xd[..., torch.from_numpy(order[:k])] = fill
        xs.append(xd.view_as(x))
    probs = torch.softmax(model(torch.cat(xs)), 1)[:, t].numpy()
    return float(np.trapezoid(probs, dx=1.0 / steps))


def random_deletion_auc(model, x, target=None, steps: int = 20, n: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    H, W = x.shape[-2:]
    return float(np.mean([deletion_auc(model, x, rng.random((H, W)), target, steps) for _ in range(n)]))


def overlay(image01: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB overlay (H x W x 3, float in [0,1]) of a heatmap on a grayscale image."""
    from matplotlib import colormaps

    rgb = np.repeat(_normalise(image01)[..., None], 3, axis=2)
    cm = colormaps["jet"](_normalise(heat))[..., :3]
    return (1 - alpha) * rgb + alpha * cm
