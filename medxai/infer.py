"""Single-image inference with input validation (DEPLOYMENT.md).

Research-only: outputs are model scores, not a diagnosis.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from . import CLASS_NAMES, __version__
from .models import load_checkpoint

NOTICE = ("RESEARCH USE ONLY. This model output is not a medical diagnosis and must not be used "
          "for clinical decisions.")
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_BYTES = 20 * 1024 * 1024
MIN_SIDE = 16
MAX_SIDE = 8192


class InputError(ValueError):
    pass


def validate_and_load(path_or_image, channels: int = 1) -> np.ndarray:
    """Return a uint8 array (H x W for ``channels=1``, H x W x 3 for ``channels=3``),
    or raise InputError with a clear message."""
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        path = str(path_or_image)
        if not os.path.isfile(path):
            raise InputError(f"file not found: {path}")
        if os.path.splitext(path)[1].lower() not in ALLOWED_EXT:
            raise InputError(f"unsupported file type; allowed: {sorted(ALLOWED_EXT)}")
        if os.path.getsize(path) > MAX_BYTES:
            raise InputError(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
        if os.path.getsize(path) == 0:
            raise InputError("empty file")
        try:
            img = Image.open(path)
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise InputError(f"could not read image: {e}") from e
    w, h = img.size
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise InputError(f"image size {w}x{h} outside allowed range [{MIN_SIDE}, {MAX_SIDE}]")
    if img.mode not in ("L", "I;16", "I", "RGB", "RGBA", "P", "LA"):
        raise InputError(f"unsupported colour mode {img.mode}")
    if channels == 3:
        if img.mode in ("I;16", "I"):
            raise InputError("16-bit grayscale image given to a colour (RGB) model")
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    if img.mode in ("RGB", "RGBA", "P", "LA"):
        rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
        if np.abs(rgb[..., 0] - rgb[..., 1]).max() > 8 or np.abs(rgb[..., 1] - rgb[..., 2]).max() > 8:
            raise InputError("colour image detected; this model expects a grayscale chest X-ray")
    if img.mode in ("I;16", "I"):
        a = np.asarray(img, dtype=np.float64)
        a = (a - a.min()) / max(a.max() - a.min(), 1) * 255
        return a.astype(np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def preprocess(gray: np.ndarray, size: int, normalizer: dict) -> torch.Tensor:
    """Centre-crop to square and resize, mirroring MedMNIST's documented preprocessing
    (centre crop + resize), then apply the training-set normalisation.

    Accepts H x W (grayscale) or H x W x 3 (RGB, per-channel normaliser); returns 1 x C x S x S.
    """
    h, w = gray.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    sq = Image.fromarray(gray[top : top + s, left : left + s])
    # The exact resampling kernel used to build MedMNIST was not verified here; external
    # images may therefore differ slightly from the training distribution (see README limitations).
    arr = np.asarray(sq.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    if arr.ndim == 3:
        x = (arr - np.float32(normalizer["mean"])) / np.float32(normalizer["std"])
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()[None]
    x = (arr - normalizer["mean"]) / normalizer["std"]
    return torch.from_numpy(x)[None, None]


class Predictor:
    def __init__(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"model checkpoint not found: {ckpt_path}. Train one with `python -m scripts.train` "
                "(weights are not distributed in this repository)."
            )
        self.model, self.meta = load_checkpoint(ckpt_path)
        self.ckpt_path = ckpt_path

    def predict(self, path_or_image, explain_method: str | None = None) -> dict:
        channels = int(self.meta.get("in_channels", 1))
        names = list(self.meta.get("class_names", CLASS_NAMES))
        norm = self.meta["normalizer"]
        img = validate_and_load(path_or_image, channels)
        x = preprocess(img, self.meta["input_size"], norm)
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0].numpy()
        out = {"scores": {c: float(v) for c, v in zip(names, p)}}
        if len(p) == 2:
            thr = float(self.meta.get("val_threshold_youden", 0.5))
            out.update({"predicted_label_at_0.5": names[int(p[1] >= 0.5)],
                        "predicted_label_at_val_threshold": names[int(p[1] >= thr)],
                        "val_threshold": thr})
        else:
            out["predicted_label_argmax"] = names[int(p.argmax())]
        out.update({
            "model": self.meta["model"], "checkpoint": os.path.basename(self.ckpt_path),
            "checkpoint_epoch": self.meta.get("epoch"), "medxai_version": __version__,
            "trained_with_version": self.meta.get("medxai_version"),
            "input_size": self.meta["input_size"], "notice": NOTICE,
        })
        if explain_method:
            from .xai import explain

            out["_heatmap"] = explain(self.model, x, explain_method)
            if channels == 3:
                out["_input01"] = (x[0].permute(1, 2, 0).numpy() * np.float32(norm["std"])
                                   + np.float32(norm["mean"]))
            else:
                out["_input01"] = x[0, 0].numpy() * norm["std"] + norm["mean"]
        return out
