import numpy as np
import pytest

from medxai.data import (ArrayDataset, Normalizer, augment_batch, load_npz, make_synthetic_npz,
                         validate_dataset)


def test_load_and_shapes(synth_npz):
    d = load_npz(synth_npz)
    assert set(d) == {"train", "val", "test"}
    x, y = d["train"]
    assert x.shape == (80, 28, 28) and x.dtype == np.uint8
    assert y.shape == (80,) and set(np.unique(y)) <= {0, 1}


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATASET_GUIDE"):
        load_npz(str(tmp_path / "nope.npz"))


def test_missing_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, train_images=np.zeros((2, 28, 28), np.uint8))
    with pytest.raises(KeyError):
        load_npz(str(p))


def test_validation_clean(synth_npz):
    rep = validate_dataset(load_npz(synth_npz), {"train": 80, "val": 30, "test": 30})
    assert rep.ok, rep.errors
    assert rep.stats["cross_split_exact_duplicates"] == {"train-val": 0, "train-test": 0, "val-test": 0}
    assert rep.stats["train"]["class_counts"] == {"0": 40, "1": 40}


def test_validation_detects_problems(synth_npz):
    d = load_npz(synth_npz)
    tx, ty = d["train"]
    vx, vy = d["val"]
    vx = vx.copy()
    vx[0] = tx[0]                      # leak a train image into val
    ty = ty.copy()
    ty[1] = 7                          # invalid label
    d["train"], d["val"] = (tx, ty), (vx, vy)
    rep = validate_dataset(d, {"train": 80, "val": 31, "test": 30})
    assert not rep.ok
    assert any("outside" in e for e in rep.errors)
    assert any("expected 31" in e for e in rep.errors)
    assert rep.stats["cross_split_exact_duplicates"]["train-val"] == 1


def test_validation_empty_split(synth_npz):
    d = load_npz(synth_npz)
    d["val"] = (np.zeros((0, 28, 28), np.uint8), np.zeros((0,), np.uint8))
    assert not validate_dataset(d).ok


def test_normalizer_fit_on_train_only(synth_npz):
    d = load_npz(synth_npz)
    n = Normalizer.fit(d["train"][0])
    z = n(d["train"][0])
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3
    assert n.mean == pytest.approx(d["train"][0].mean() / 255, rel=1e-6)


def test_dataset_and_augment(synth_npz):
    import torch

    d = load_npz(synth_npz)
    ds = ArrayDataset(*d["val"], Normalizer.fit(d["train"][0]))
    x, y, i = ds[3]
    assert x.shape == (1, 28, 28) and int(i) == 3
    g = torch.Generator().manual_seed(0)
    a1 = augment_batch(ds.x[:8], g)
    a2 = augment_batch(ds.x[:8], torch.Generator().manual_seed(0))
    assert a1.shape == ds.x[:8].shape and torch.allclose(a1, a2)


def test_synthetic_deterministic(tmp_path):
    a = load_npz(make_synthetic_npz(str(tmp_path / "a.npz"), seed=3))
    b = load_npz(make_synthetic_npz(str(tmp_path / "b.npz"), seed=3))
    assert np.array_equal(a["test"][0], b["test"][0])
