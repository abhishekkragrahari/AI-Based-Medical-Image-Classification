"""E10/M1: multi-class, RGB and dataset-registry support. Binary behaviour is covered by the
existing tests, which must keep passing unchanged."""

import hashlib
import json

import numpy as np
import pytest
import torch

from medxai.data import (DATASETS, ArrayDataset, Normalizer, dataset_url, load_npz,
                         make_synthetic_npz, npz_filename, validate_dataset)
from medxai.metrics import binary_metrics, multiclass_metrics
from medxai.models import load_checkpoint


@pytest.fixture(scope="module")
def rgb_npz(tmp_path_factory):
    return make_synthetic_npz(str(tmp_path_factory.mktemp("rgb") / "rgb.npz"), counts=(96, 32, 32),
                              n_classes=4, channels=3)


def test_registry_matches_medmnist_conventions():
    assert npz_filename(28, "bloodmnist") == "bloodmnist.npz"
    assert npz_filename(224, "breastmnist") == "breastmnist_224.npz"
    assert dataset_url("bloodmnist", 64).endswith("/bloodmnist_64.npz?download=1")
    for flag, info in DATASETS.items():
        assert set(info["md5"]) == {28, 64, 128, 224}
        assert set(info["counts"]) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="unknown dataset"):
        npz_filename(28, "dermamnist")


def test_binary_synthetic_unchanged(tmp_path):
    """Default synthetic data is byte-identical to the generator before E10 (fixed digest)."""
    p = make_synthetic_npz(str(tmp_path / "s.npz"), counts=(20, 10, 10))
    d = load_npz(p)
    h = hashlib.sha1(d["train"][0].tobytes() + d["train"][1].tobytes()).hexdigest()
    assert h == _binary_digest()


def _binary_digest():
    # Recomputed with the pre-E10 generator logic: same RNG call order, labels i % 2, one blob.
    rng = np.random.default_rng(0)
    size, n = 28, 20
    yy, xx = np.mgrid[0:size, 0:size]
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8)
    rng.shuffle(labels)
    imgs = rng.normal(100, 20, size=(n, size, size))
    for k in np.where(labels == 1)[0]:
        cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
        imgs[k] += 90 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 10) ** 2))
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    return hashlib.sha1(imgs.tobytes() + labels.reshape(-1).tobytes()).hexdigest()


def test_rgb_multiclass_validation_and_tensors(rgb_npz):
    d = load_npz(rgb_npz)
    x, y = d["train"]
    assert x.shape == (96, 28, 28, 3) and set(np.unique(y)) == {0, 1, 2, 3}
    rep = validate_dataset(d, {"train": 96, "val": 32, "test": 32}, n_classes=4)
    assert rep.ok, rep.errors
    assert set(rep.stats["train"]["class_counts"]) == {"0", "1", "2", "3"}
    assert "cross_split_near_duplicate_candidates" in rep.stats
    # Labels outside the declared classes are errors.
    assert not validate_dataset(d, n_classes=3).ok

    norm = Normalizer.fit(x)
    assert isinstance(norm.mean, list) and len(norm.mean) == 3 and len(norm.std) == 3
    ds = ArrayDataset(x, y, norm)
    assert ds.x.shape == (96, 3, 28, 28)
    # Per-channel normalisation: each channel has ~zero mean, unit std on the training split.
    assert torch.allclose(ds.x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(ds.x.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-2)
    json.dumps(norm.to_dict())  # stays JSON-serialisable for checkpoints


def test_grayscale_normalizer_unchanged(synth_npz):
    x = load_npz(synth_npz)["train"][0]
    n = Normalizer.fit(x)
    assert isinstance(n.mean, float) and isinstance(n.std, float)
    assert ArrayDataset(x, np.zeros(len(x)), n).x.shape == (len(x), 1, 28, 28)


def test_multiclass_metrics_hand_computed():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .8, .1], [.1, .6, .3], [.1, .1, .8], [.3, .3, .4]])
    m = multiclass_metrics(y, p, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(5 / 6)  # only row 1 (true 0, predicted 1) is wrong
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert m["n_auroc_classes"] == 3 and 0.5 < m["auroc"] <= 1.0
    perfect = multiclass_metrics(y, np.eye(3)[y])
    assert perfect["auroc"] == 1.0 and perfect["ece"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        multiclass_metrics(y, p[:, :2].ravel())


def test_binary_metrics_class_names_only_rename_keys():
    y, s = np.array([0, 1, 1, 0]), np.array([.2, .9, .4, .6])
    a, b = binary_metrics(y, s), binary_metrics(y, s, class_names=("malignant", "normal_benign"))
    assert set(a["per_class"]) == {"normal", "pneumonia"}
    assert set(b["per_class"]) == {"malignant", "normal_benign"}
    assert a["per_class"]["pneumonia"] == b["per_class"]["normal_benign"]
    assert {k: v for k, v in a.items() if k != "per_class"} == {k: v for k, v in b.items() if k != "per_class"}


def test_train_rgb_multiclass(rgb_npz, tmp_path):
    from medxai.train import TrainConfig, predict_proba_all, train

    cfg = TrainConfig(data_path=rgb_npz, dataset="bloodmnist", model="small_cnn",
                      model_kwargs={"width": 8}, epochs=3, batch_size=32, out_dir=str(tmp_path),
                      run_name="mc", class_weighting=True)
    s = train(cfg, verbose=False)
    assert s["n_classes"] == 4 and "val_at_argmax" in s and "val_at_0.5" not in s
    # 4 classes in the data != 8 BloodMNIST names, so generic names are used.
    assert set(s["val_at_argmax"]["per_class"]) == {"0", "1", "2", "3"}
    assert np.load(f"{tmp_path}/mc/val_scores.npy").shape == (32, 4)
    model, meta = load_checkpoint(f"{tmp_path}/mc/best.pt")
    assert meta["n_classes"] == 4 and meta["in_channels"] == 3
    x = torch.zeros(2, 3, 28, 28)
    p = predict_proba_all(model, x)
    assert p.shape == (2, 4) and np.allclose(p.sum(1), 1, atol=1e-5)


@pytest.mark.parametrize("model_name", ["small_cnn", "resnet18"])
def test_train_breast_like_binary_with_names(synth_npz, tmp_path, model_name):
    from medxai.train import TrainConfig, train

    cfg = TrainConfig(data_path=synth_npz, dataset="breastmnist", model=model_name,
                      model_kwargs={"width": 8} if model_name == "small_cnn" else {}, epochs=1,
                      batch_size=32, out_dir=str(tmp_path), run_name="b", max_train_samples=40)
    s = train(cfg, verbose=False)
    assert set(s["val_at_0.5"]["per_class"]) == {"malignant", "normal_benign"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_shapes(rgb_npz, tmp_path):
    from medxai.train import TrainConfig, train

    s = train(TrainConfig(data_path=rgb_npz, model="small_cnn", model_kwargs={"width": 8}, epochs=1,
                          batch_size=32, out_dir=str(tmp_path), run_name="gpu", device="cuda"),
              verbose=False)
    assert s["n_classes"] == 4
    _, meta = load_checkpoint(f"{tmp_path}/gpu/best.pt")  # saved from CUDA, loads on CPU
    assert meta["config"]["device"] == "cuda"


def test_unknown_device_rejected():
    from medxai.train import resolve_device

    with pytest.raises(ValueError):
        resolve_device("tpu")
