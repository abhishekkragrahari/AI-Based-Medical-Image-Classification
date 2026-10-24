"""Data loading, validation and preprocessing for MedMNIST (v2 / MedMNIST+) 2D datasets.

PneumoniaMNIST is the default; BreastMNIST and BloodMNIST are registered for E10.
The ``.npz`` layout follows the official MedMNIST release
(https://github.com/MedMNIST/MedMNIST, data on Zenodo record 10519652):
keys ``{train,val,test}_images`` (uint8, N x H x W, or N x H x W x 3 for RGB) and
``{train,val,test}_labels`` (N x 1, values in {0, ..., n_classes - 1}).

Normalisation statistics are fitted on the training split only.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")

# MD5 checksums published in the official MedMNIST v3.0.2 ``info.py``.
OFFICIAL_MD5 = {
    28: "28209eda62fecd6e6a2d98b1501bb15f",
    64: "8f4eceb4ccffa70c672198ea285246c6",
    128: "05b46931834c231683c68f40c47b2971",
    224: "d6a3c71de1b945ea11211b03746c1fe1",
}
OFFICIAL_COUNTS = {"train": 4708, "val": 524, "test": 624}
ZENODO_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist{flag}.npz?download=1"

# Registered datasets: MD5s, counts and label names copied from the official MedMNIST
# v3.0.2 ``medmnist.INFO`` (checked 2026-10-24). Licences: all CC BY 4.0.
DATASETS = {
    "pneumoniamnist": {
        "md5": OFFICIAL_MD5, "counts": OFFICIAL_COUNTS, "n_channels": 1,
        "classes": ["normal", "pneumonia"],
    },
    "breastmnist": {
        "md5": {28: "750601b1f35ba3300ea97c75c52ff8f6", 64: "742edef2a1fd1524b2efff4bd7ba9364",
                128: "363e4b3f8d712e9b5de15470a2aaadf1", 224: "b56378a6eefa9fed602bb16d192d4c8b"},
        "counts": {"train": 546, "val": 78, "test": 156}, "n_channels": 1,
        # Note the label order: 0 = malignant, 1 = normal/benign (official MedMNIST labels).
        "classes": ["malignant", "normal_benign"],
    },
    "bloodmnist": {
        "md5": {28: "7053d0359d879ad8a5505303e11de1dc", 64: "2b94928a2ae4916078ca51e05b6b800b",
                128: "adace1e0ed228fccda1f39692059dd4c", 224: "b718ff6835fcbdb22ba9eacccd7b2601"},
        "counts": {"train": 11959, "val": 1712, "test": 3421}, "n_channels": 3,
        "classes": ["basophil", "eosinophil", "erythroblast", "immature_granulocytes", "lymphocyte",
                    "monocyte", "neutrophil", "platelet"],
    },
}


def dataset_info(flag: str) -> dict:
    if flag not in DATASETS:
        raise ValueError(f"unknown dataset '{flag}'; choose from {sorted(DATASETS)}")
    return DATASETS[flag]


def npz_filename(size: int = 28, flag: str = "pneumoniamnist") -> str:
    dataset_info(flag)
    return f"{flag}.npz" if size == 28 else f"{flag}_{size}.npz"


def dataset_url(flag: str = "pneumoniamnist", size: int = 28) -> str:
    return f"https://zenodo.org/records/10519652/files/{npz_filename(size, flag)}?download=1"


def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load an npz file into ``{split: (images, labels)}`` with labels flattened."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Download it from "
            f"{ZENODO_URL.format(flag='')} (see DATASET_GUIDE.md)."
        )
    # allow_pickle=False: never execute pickled objects from a downloaded file.
    with np.load(path, allow_pickle=False) as npz:
        out = {}
        for split in SPLITS:
            ik, lk = f"{split}_images", f"{split}_labels"
            if ik not in npz or lk not in npz:
                raise KeyError(f"{path} is missing '{ik}' or '{lk}'")
            out[split] = (np.asarray(npz[ik]), np.asarray(npz[lk]).reshape(-1))
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def _hashes(images: np.ndarray) -> list[str]:
    return [hashlib.sha1(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images]


def train_duplicate_mask(train_images: np.ndarray, images: np.ndarray) -> np.ndarray:
    """True where an image is a byte-identical copy of some training image."""
    seen = set(_hashes(train_images))
    return np.array([h in seen for h in _hashes(images)], dtype=bool)


def _near_dup_keys(images: np.ndarray, side: int = 8) -> list[bytes]:
    """Coarse perceptual key: block-mean downsample to side x side, then 4-bit quantise.

    Collisions flag *candidate* near-duplicates for manual review only.
    """
    if images.ndim == 4:  # RGB: compare luminance-like channel mean
        images = images.mean(axis=3)
    n, h, w = images.shape[:3]
    bh, bw = h // side, w // side
    x = images[:, : bh * side, : bw * side].astype(np.float32)
    x = x.reshape(n, side, bh, side, bw).mean(axis=(2, 4))
    x = (x - x.min(axis=(1, 2), keepdims=True))
    rng = x.max(axis=(1, 2), keepdims=True)
    x = np.where(rng > 0, x / np.maximum(rng, 1e-8), 0.0)
    q = np.clip((x * 15).round(), 0, 15).astype(np.uint8)
    return [row.tobytes() for row in q]


def validate_dataset(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    expected_counts: dict[str, int] | None = None,
    near_duplicates: bool = True,
    n_classes: int = 2,
) -> ValidationReport:
    """Integrity and leakage checks described in DATASET_GUIDE.md / TESTING.md."""
    rep = ValidationReport()
    classes = list(range(n_classes))
    split_hashes = {}
    split_near = {}
    for split in SPLITS:
        if split not in data:
            rep.error(f"missing split '{split}'")
            continue
        imgs, labels = data[split]
        s = {"n": int(len(imgs))}
        if len(imgs) == 0:
            rep.error(f"{split}: empty split")
            rep.stats[split] = s
            continue
        if len(imgs) != len(labels):
            rep.error(f"{split}: {len(imgs)} images but {len(labels)} labels")
        if not (imgs.ndim == 3 or (imgs.ndim == 4 and imgs.shape[3] == 3)):
            rep.error(f"{split}: expected grayscale N x H x W or RGB N x H x W x 3, got shape {imgs.shape}")
        if imgs.dtype != np.uint8:
            rep.error(f"{split}: expected uint8 images, got {imgs.dtype}")
        bad = ~np.isin(labels, classes)
        if bad.any():
            rep.error(f"{split}: {int(bad.sum())} labels outside {{{','.join(map(str, classes))}}}")
        s["shape"] = list(imgs.shape[1:])
        s["class_counts"] = {str(c): int((labels == c).sum()) for c in classes}
        s["pixel_min"], s["pixel_max"] = int(imgs.min()), int(imgs.max())
        s["pixel_mean"] = float(imgs.mean())
        flat_std = imgs.reshape(len(imgs), -1).std(axis=1)
        s["n_constant_images"] = int((flat_std == 0).sum())
        if s["n_constant_images"]:
            rep.warnings.append(f"{split}: {s['n_constant_images']} constant (blank) images")
        if expected_counts and split in expected_counts and len(imgs) != expected_counts[split]:
            rep.error(f"{split}: expected {expected_counts[split]} samples, found {len(imgs)}")

        h = _hashes(imgs)
        split_hashes[split] = h
        s["n_within_split_exact_duplicates"] = len(h) - len(set(h))
        # Duplicates with conflicting labels are a label-quality problem.
        by_hash: dict[str, set] = {}
        for hh, lab in zip(h, labels):
            by_hash.setdefault(hh, set()).add(int(lab))
        s["n_duplicate_groups_conflicting_labels"] = sum(len(v) > 1 for v in by_hash.values())
        if s["n_within_split_exact_duplicates"]:
            rep.warnings.append(
                f"{split}: {s['n_within_split_exact_duplicates']} exact duplicate images within split"
            )
        if near_duplicates and imgs.ndim in (3, 4):
            split_near[split] = _near_dup_keys(imgs)
        rep.stats[split] = s

    cross = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if a in split_hashes and b in split_hashes:
                n = len(set(split_hashes[a]) & set(split_hashes[b]))
                cross[f"{a}-{b}"] = n
                if n:
                    rep.warnings.append(f"{n} exact duplicate images shared between {a} and {b}")
    rep.stats["cross_split_exact_duplicates"] = cross
    if split_near:
        near = {}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1 :]:
                if a in split_near and b in split_near:
                    sb = set(split_near[b])
                    near[f"{a}-{b}"] = int(sum(k in sb for k in split_near[a]))
        rep.stats["cross_split_near_duplicate_candidates"] = near
    rep.stats["patient_ids_available"] = False
    rep.warnings.append(
        "No patient identifiers in the MedMNIST release: patient-level overlap between "
        "train and val cannot be checked (the source train set was split 9:1 by image)."
    )
    return rep


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class Normalizer:
    """Intensity normalisation; fit on training images only.

    Grayscale: one global mean/std (floats). RGB (N x H x W x 3): per-channel lists.
    """

    mean: float | list = 0.5
    std: float | list = 0.25

    @classmethod
    def fit(cls, train_images: np.ndarray) -> "Normalizer":
        x = train_images.astype(np.float64) / 255.0
        if x.ndim == 4:
            std = x.std(axis=(0, 1, 2))
            return cls(mean=[float(v) for v in x.mean(axis=(0, 1, 2))],
                       std=[float(v) if v > 0 else 1.0 for v in std])
        std = float(x.std())
        return cls(mean=float(x.mean()), std=std if std > 0 else 1.0)

    def __call__(self, images_uint8: np.ndarray) -> np.ndarray:
        x = images_uint8.astype(np.float32) / 255.0
        if isinstance(self.mean, list):
            return (x - np.float32(self.mean)) / np.float32(self.std)
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def augment_batch(x: torch.Tensor, generator: torch.Generator, max_shift: int = 2,
                  max_rot_deg: float = 10.0, return_theta: bool = False):
    """Mild train-only augmentation: small random rotation and translation.

    Horizontal flips are deliberately *not* used: chest X-rays have anatomical
    left/right asymmetry (heart position), which flipping would distort.
    With ``return_theta`` the affine matrices (n x 2 x 3, ``affine_grid`` convention) are
    returned too, so maps can be warped the same way; the random draws are identical.
    """
    n = x.shape[0]
    ang = (torch.rand(n, generator=generator) * 2 - 1) * (max_rot_deg * np.pi / 180)
    h, w = x.shape[-2:]
    tx = (torch.randint(-max_shift, max_shift + 1, (n,), generator=generator).float() * 2 / w)
    ty = (torch.randint(-max_shift, max_shift + 1, (n,), generator=generator).float() * 2 / h)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    # Pad with the per-image minimum so borders look like background, not a bright frame.
    fill = x.amin(dim=(1, 2, 3), keepdim=True)
    out = torch.nn.functional.grid_sample(x - fill, grid, align_corners=False, padding_mode="zeros")
    return (out + fill, theta) if return_theta else out + fill


def to_model_input(images_uint8: np.ndarray, normalizer: dict) -> torch.Tensor:
    """N x C x H x W float tensor from uint8 images and a checkpoint's normaliser dict
    (``{"mean", "std"}`` as floats for grayscale or per-channel lists for RGB)."""
    norm = Normalizer(mean=normalizer["mean"], std=normalizer["std"])
    x = torch.from_numpy(norm(images_uint8))
    return x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x.unsqueeze(1)


class ArrayDataset(Dataset):
    """Tensor dataset over normalised images: returns (C x H x W float tensor, int label, index)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, normalizer: Normalizer):
        x = torch.from_numpy(normalizer(images))
        self.x = x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x.unsqueeze(1)
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


def make_synthetic_npz(path: str, size: int = 28, counts=(60, 20, 20), seed: int = 0,
                       n_classes: int = 2, channels: int = 1) -> str:
    """Write a small synthetic MedMNIST-format file for tests (no real patient data).

    Class c >= 1 images contain a bright blob (larger for higher c) in a random central
    position so that a model can learn a non-trivial, localisable signal. With
    ``channels=3`` the image is repeated over RGB and class c tints channel c % 3.
    The default (binary, grayscale) output is unchanged from earlier versions.
    """
    rng = np.random.default_rng(seed)
    arrays = {}
    yy, xx = np.mgrid[0:size, 0:size]
    for split, n in zip(SPLITS, counts):
        labels = np.array([i % n_classes for i in range(n)], dtype=np.uint8)
        rng.shuffle(labels)
        imgs = rng.normal(100, 20, size=(n, size, size))
        for k in np.where(labels >= 1)[0]:
            cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
            sigma = size / 10 * (1 + 0.5 * (int(labels[k]) - 1))
            imgs[k] += 90 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        if channels == 3:
            imgs = np.repeat(imgs[..., None], 3, axis=3)
            for k in range(n):
                imgs[k, ..., int(labels[k]) % 3] += 30
        arrays[f"{split}_images"] = np.clip(imgs, 0, 255).astype(np.uint8)
        arrays[f"{split}_labels"] = labels.reshape(-1, 1)
    np.savez_compressed(path, **arrays)
    return path
