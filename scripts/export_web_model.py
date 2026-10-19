"""Export a SmallCNN checkpoint to a JSON file a web page can run without a server.

BatchNorm is folded into the preceding convolution, so the browser only needs
conv3x3 + bias, ReLU, 2x2 max-pool, global average pooling and one linear layer.
Weights are float32, base64-encoded. A few validation images (never test images)
are included as samples. ``reference_forward`` is the NumPy version of what the
page computes; the tests check it against PyTorch.

Usage: python -m scripts.export_web_model --run runs/small_cnn_s0 [--data data/pneumoniamnist.npz]
       [--out web_model.json] [--n-samples 6]
"""

import argparse
import base64
import json
import os
import sys

import numpy as np

from medxai.models import SmallCNN, load_checkpoint

LAYERS = [f"block{b}.{i}" for b in (1, 2, 3) for i in (0, 3)]  # conv indices inside each block


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _finite(o):
    """Replace NaN/inf (e.g. undefined precision) with None so the output is strict JSON."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o


def fold_layers(model: SmallCNN) -> list[dict]:
    """Return [{w: (cout, cin, 3, 3), b: (cout,)}] with BatchNorm folded in."""
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    out = []
    for name in LAYERS:
        blk, idx = name.split(".")
        bn = f"{blk}.{int(idx) + 1}"
        scale = sd[f"{bn}.weight"] / np.sqrt(sd[f"{bn}.running_var"] + model.get_submodule(bn).eps)
        out.append({"w": sd[f"{name}.weight"] * scale[:, None, None, None],
                    "b": sd[f"{bn}.bias"] - sd[f"{bn}.running_mean"] * scale})
    return out


def _conv3x3(x, w, b):
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, i:i + h, j:j + wd] for i in range(3) for j in range(3)], axis=1)  # c,9,h,w
    return np.einsum("ock,ckhw->ohw", w.reshape(w.shape[0], c, 9), cols) + b[:, None, None]


def reference_forward(export: dict, x: np.ndarray) -> np.ndarray:
    """Logits for one normalised image x of shape (H, W), using the exported JSON."""
    h = x[None].astype(np.float64)
    for i, layer in enumerate(export["layers"]):
        w = np.frombuffer(base64.b64decode(layer["w"]), np.float32).reshape(layer["shape"])
        b = np.frombuffer(base64.b64decode(layer["b"]), np.float32)
        h = np.maximum(_conv3x3(h, w, b), 0)
        if i in (1, 3):
            c, hh, ww = h.shape
            h = h[:, :hh // 2 * 2, :ww // 2 * 2].reshape(c, hh // 2, 2, ww // 2, 2).max(axis=(2, 4))
    lw = np.frombuffer(base64.b64decode(export["linear"]["w"]), np.float32).reshape(export["linear"]["shape"])
    lb = np.frombuffer(base64.b64decode(export["linear"]["b"]), np.float32)
    return lw @ h.mean(axis=(1, 2)) + lb


def export(run_dir: str, data_path: str | None = None, n_samples: int = 6) -> dict:
    model, meta = load_checkpoint(os.path.join(run_dir, "best.pt"))
    if not isinstance(model, SmallCNN):
        raise ValueError("only small_cnn checkpoints can be exported for the browser")
    lin = model.head[-1]
    out = {
        "format": "medxai-web-1",
        "model": meta["model"], "model_kwargs": meta.get("model_kwargs", {}),
        "input_size": meta["input_size"], "normalizer": meta["normalizer"],
        "val_threshold_youden": meta.get("val_threshold_youden"),
        "class_names": ["normal", "pneumonia"],
        "layers": [{"shape": list(l["w"].shape), "w": _b64(l["w"], "<f4"), "b": _b64(l["b"], "<f4")}
                   for l in fold_layers(model)],
        "linear": {"shape": list(lin.weight.shape), "w": _b64(lin.weight.detach().numpy(), "<f4"),
                   "b": _b64(lin.bias.detach().numpy(), "<f4")},
        "samples": [],
    }
    summary = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            s = json.load(f)
        out["run"] = _finite({k: s.get(k) for k in ("best_epoch", "n_params", "data_md5", "val_at_0.5")})
    if data_path and n_samples > 0:
        from medxai.data import load_npz

        images, labels = load_npz(data_path)["val"]
        rng = np.random.default_rng(0)
        for cls in (0, 1):
            for i in rng.choice(np.flatnonzero(labels == cls), n_samples // 2, replace=False):
                out["samples"].append({"split": "val", "index": int(i), "label": int(labels[i]),
                                       "h": int(images[i].shape[0]), "w": int(images[i].shape[1]),
                                       "pixels": _b64(images[i], np.uint8)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--data", default=None, help="npz to take validation samples from")
    ap.add_argument("--out", default="web_model.json")
    ap.add_argument("--n-samples", type=int, default=6)
    a = ap.parse_args(argv)
    e = export(a.run, a.data, a.n_samples)
    with open(a.out, "w") as f:
        json.dump(e, f, allow_nan=False)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB, {len(e['samples'])} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
