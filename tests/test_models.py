import pytest
import torch

from medxai.models import build_model, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("name,size", [("small_cnn", 28), ("resnet18", 28), ("small_cnn", 64)])
def test_output_shape(name, size):
    m = build_model(name).eval()
    with torch.no_grad():
        out = m(torch.randn(3, 1, size, size))
    assert out.shape == (3, 2)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("vit_huge")


def test_checkpoint_roundtrip(tmp_path):
    m = build_model("small_cnn", width=8).eval()
    meta = {"model": "small_cnn", "model_kwargs": {"width": 8}, "input_size": 28,
            "normalizer": {"mean": 0.5, "std": 0.2}}
    p = tmp_path / "c.pt"
    save_checkpoint(str(p), m, meta)
    m2, meta2 = load_checkpoint(str(p))
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))
    assert meta2["input_size"] == 28


def test_rejects_non_medxai_checkpoint(tmp_path):
    p = tmp_path / "x.pt"
    torch.save({"foo": torch.zeros(1)}, p)
    with pytest.raises(ValueError):
        load_checkpoint(str(p))


def test_rejects_pickled_code(tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    p = tmp_path / "evil.pt"
    with open(p, "wb") as f:
        pickle.dump({"state_dict": Evil(), "meta": {}}, f)
    with pytest.raises(Exception):
        load_checkpoint(str(p))
