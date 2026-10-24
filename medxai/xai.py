"""Explainability methods and explanation-quality checks (see XAI_PLAN.md).

Methods: Grad-CAM, Integrated Gradients, occlusion sensitivity.
Checks:  border-mass (shortcut reliance), cross-method/seed rank agreement,
         perturbation stability, deletion/insertion-curve faithfulness with several
         fill strategies and pixel- or patch-wise removal, cascading model randomisation.

All functions take a single image x of shape 1 x C x H x W (grayscale or RGB) on the
model's device. Sizes that matter for resolution comparisons (occlusion patch, deletion
patch, blur sigma) scale with the image side; at 28 px they equal the earlier fixed values.

Heatmaps show which pixels influence the model's output; they do not show that
the model reasons in a clinically valid way.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

FILLS = ("zero", "blur", "noise")
UNITS = ("pixel", "patch")


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def _side(x: torch.Tensor) -> int:
    return int(min(x.shape[-2:]))


def _chunk(side: int, base: int = 256) -> int:
    """Forward-pass batch size that keeps memory roughly constant across resolutions."""
    return max(4, int(base * (28 / side) ** 2))


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
    return _normalise(cam.cpu().numpy())


def integrated_gradients(model: nn.Module, x: torch.Tensor, target: int | None = None,
                         baseline: torch.Tensor | None = None, steps: int = 32,
                         batch_size: int | None = None) -> np.ndarray:
    """Integrated Gradients (Sundararajan et al., ICML 2017), Riemann midpoint rule.

    Baseline defaults to the all-zeros tensor in *normalised* space, i.e. the
    training-set mean intensity. Returns |attribution| summed over channels, normalised
    to [0, 1]. The path is evaluated in chunks of ``batch_size`` (default: scaled to size).
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    with torch.no_grad():
        t = int(model(x).argmax(1)) if target is None else int(target)
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps
    bs = batch_size or min(steps, _chunk(_side(x), 32))
    gsum = torch.zeros_like(x)
    for k in range(0, steps, bs):
        path = (baseline + alphas[k : k + bs].view(-1, 1, 1, 1) * (x - baseline)).requires_grad_(True)
        (g,) = torch.autograd.grad(model(path)[:, t].sum(), path)
        gsum += g.sum(0, keepdim=True)
    attr = ((x - baseline) * gsum / steps)[0].sum(0)
    return _normalise(attr.abs().detach().cpu().numpy())


def reference_image(x: torch.Tensor, fill: float | str | torch.Tensor = 0.0, seed: int = 0) -> torch.Tensor:
    """Values that replace removed pixels (deletion) or form the start image (insertion).

    float: constant in normalised space (0.0 = training-set mean; earlier default).
    ``"zero"``: same as 0.0. ``"blur"``: Gaussian-blurred copy of x, sigma = side/14.
    ``"noise"``: N(0, 1) in normalised space (training-set marginal scale), seeded.
    A tensor is used as given (same shape as x).
    """
    if isinstance(fill, torch.Tensor):
        return fill.to(x.device, x.dtype).expand_as(x)
    if isinstance(fill, (int, float)) and not isinstance(fill, bool):
        return torch.full_like(x, float(fill))
    if fill == "zero":
        return torch.zeros_like(x)
    if fill == "blur":
        return gaussian_blur(x, sigma=_side(x) / 14)
    if fill == "noise":
        g = torch.Generator().manual_seed(seed)
        return torch.randn(x.shape, generator=g, dtype=x.dtype).to(x.device)
    raise ValueError(f"unknown fill {fill!r}; use a number, a tensor or one of {FILLS}")


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur per channel with reflect padding."""
    r = max(1, int(math.ceil(3 * sigma)))
    r = min(r, min(x.shape[-2:]) - 1)  # reflect padding must be smaller than the image
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    y = F.pad(x, (r, r, r, r), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


@torch.no_grad()
def occlusion(model: nn.Module, x: torch.Tensor, target: int | None = None, patch: int | None = None,
              stride: int | None = None, fill: float | str = 0.0) -> np.ndarray:
    """Occlusion sensitivity (Zeiler & Fergus, ECCV 2014): drop in target probability
    when a patch is replaced by ``fill`` (0 = training mean in normalised space).

    Patch and stride default to side/7 and side/14 (4 and 2 at 28 px; 32 and 16 at 224 px),
    so the occluded *fraction* of the image is the same at every resolution.
    """
    model.eval()
    side = _side(x)
    patch = patch or max(1, round(side / 7))
    stride = stride or max(1, round(side / 14))
    ref = reference_image(x, fill)
    p0 = torch.softmax(model(x), 1)[0]
    t = int(p0.argmax()) if target is None else int(target)
    H, W = x.shape[-2:]
    heat = np.zeros((H, W))
    cnt = np.zeros((H, W))
    batch, coords = [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xo = x.clone()
            xo[..., i : i + patch, j : j + patch] = ref[..., i : i + patch, j : j + patch]
            batch.append(xo)
            coords.append((i, j))
    bs = _chunk(side)
    probs = torch.cat([torch.softmax(model(torch.cat(batch[k : k + bs])), 1)[:, t]
                       for k in range(0, len(batch), bs)]).cpu().numpy()
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
    vals = [rank_agreement(ref, explain(model, x + sigma * torch.randn(x.shape, generator=g).to(x.device),
                                        method, target=t, **kw)) for _ in range(n)]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _block_ids(H: int, W: int, p: int) -> np.ndarray:
    """H x W map of p x p block indices (row-major); p = 1 gives the pixel index."""
    nbw = -(-W // p)
    return (np.arange(H)[:, None] // p) * nbw + (np.arange(W)[None, :] // p)


@torch.no_grad()
def perturbation_auc(model, x, heat: np.ndarray, target: int | None = None, steps: int = 20,
                     fill: float | str | torch.Tensor = 0.0, unit: str = "pixel",
                     mode: str = "deletion") -> float:
    """Area under the target-probability curve while the most-attributed units are
    removed (``mode="deletion"``: start from x, replace by the fill reference; lower = more
    faithful) or added back (``"insertion"``: start from the reference, restore x; higher =
    more faithful). Petsiuk et al. (BMVC 2018). ``unit="patch"`` ranks and moves side/14
    patches (mean attribution per patch) instead of single pixels.
    """
    if mode not in ("deletion", "insertion"):
        raise ValueError(f"unknown mode '{mode}'")
    if unit not in UNITS:
        raise ValueError(f"unknown unit '{unit}'; choose from {UNITS}")
    model.eval()
    t = int(model(x).argmax(1)) if target is None else int(target)
    H, W = x.shape[-2:]
    p = 1 if unit == "pixel" else max(1, round(_side(x) / 14))
    ids = _block_ids(H, W, p).ravel()
    nb = int(ids.max()) + 1
    score = np.bincount(ids, weights=np.ravel(heat), minlength=nb) / np.bincount(ids, minlength=nb)
    order = np.argsort(-score)
    rank = np.empty(nb, dtype=np.int64)
    rank[order] = np.arange(nb)
    pix_rank = torch.from_numpy(rank[ids].reshape(H, W)).to(x.device)
    ref = reference_image(x, fill)
    xs = []
    for s in range(steps + 1):
        k = int(round(nb * s / steps))
        moved = pix_rank < k  # the k most-attributed units
        xs.append(torch.where(moved, ref, x) if mode == "deletion" else torch.where(moved, x, ref))
    bs = _chunk(_side(x))
    probs = torch.cat([torch.softmax(model(torch.cat(xs[k : k + bs])), 1)[:, t]
                       for k in range(0, len(xs), bs)]).cpu().numpy()
    return float(np.trapezoid(probs, dx=1.0 / steps))


def deletion_auc(model, x, heat: np.ndarray, target: int | None = None, steps: int = 20,
                 fill: float | str | torch.Tensor = 0.0, unit: str = "pixel") -> float:
    """Deletion AUC (lower = more faithful). Compare against :func:`random_deletion_auc`."""
    return perturbation_auc(model, x, heat, target, steps, fill, unit, "deletion")


def insertion_auc(model, x, heat: np.ndarray, target: int | None = None, steps: int = 20,
                  fill: float | str | torch.Tensor = 0.0, unit: str = "pixel") -> float:
    """Insertion AUC (higher = more faithful). Compare against :func:`random_insertion_auc`."""
    return perturbation_auc(model, x, heat, target, steps, fill, unit, "insertion")


def random_perturbation_auc(model, x, target=None, steps: int = 20, n: int = 5, seed: int = 0,
                            fill: float | str | torch.Tensor = 0.0, unit: str = "pixel",
                            mode: str = "deletion") -> float:
    """Mean curve AUC over ``n`` random attribution maps (the chance reference)."""
    rng = np.random.default_rng(seed)
    H, W = x.shape[-2:]
    return float(np.mean([perturbation_auc(model, x, rng.random((H, W)), target, steps, fill, unit, mode)
                          for _ in range(n)]))


def random_deletion_auc(model, x, target=None, steps: int = 20, n: int = 5, seed: int = 0,
                        fill: float | str | torch.Tensor = 0.0, unit: str = "pixel") -> float:
    return random_perturbation_auc(model, x, target, steps, n, seed, fill, unit, "deletion")


def random_insertion_auc(model, x, target=None, steps: int = 20, n: int = 5, seed: int = 0,
                         fill: float | str | torch.Tensor = 0.0, unit: str = "pixel") -> float:
    return random_perturbation_auc(model, x, target, steps, n, seed, fill, unit, "insertion")


def randomisation_stages(model: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Parameterised layers grouped by top-level block, ordered output -> input.

    A wrapper with a single child (e.g. ``ResNet18Small.net``) is looked through.
    """
    root, prefix = model, ""
    children = list(model.named_children())
    while len(children) == 1:
        prefix += children[0][0] + "."
        root = children[0][1]
        children = list(root.named_children())
    stages = []
    for name, child in children:
        mods = [m for m in child.modules() if hasattr(m, "reset_parameters")]
        if mods:
            stages.append((prefix + name, mods))
    return stages[::-1]


def model_randomisation_test(model: nn.Module, x: torch.Tensor, method: str, target: int | None = None,
                             seed: int = 0, **kw) -> list[dict]:
    """Cascading model-parameter randomisation (Adebayo et al., NeurIPS 2018).

    Re-initialises blocks from the output towards the input, one more block per stage, and
    reports the Spearman agreement between each map and the trained model's map for the
    same target class. A method whose maps barely change is insensitive to the learned
    weights and fails the sanity check. ``model`` itself is not modified.
    """
    with torch.no_grad():
        t = int(model(x).argmax(1)) if target is None else int(target)
    ref = explain(model, x, method, target=t, **kw)
    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    out = []
    for name, mods in randomisation_stages(rnd):
        for m in mods:
            m.reset_parameters()
        h = explain(rnd, x, method, target=t, **kw)
        out.append({"randomised_through": name, "spearman": rank_agreement(ref, h),
                    "top10pct_iou": topk_iou(ref, h, 0.1)})
    return out


def overlay(image01: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB overlay (H x W x 3, float in [0,1]) of a heatmap on a grayscale or RGB image."""
    from matplotlib import colormaps

    img = np.asarray(image01, dtype=float)
    rgb = _normalise(img) if img.ndim == 3 else np.repeat(_normalise(img)[..., None], 3, axis=2)
    cm = colormaps["jet"](_normalise(heat))[..., :3]
    return (1 - alpha) * rgb + alpha * cm
