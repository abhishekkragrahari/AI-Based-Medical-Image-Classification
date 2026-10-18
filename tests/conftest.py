import numpy as np
import pytest
import torch

from medxai.data import make_synthetic_npz

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def synth_npz(tmp_path_factory):
    return make_synthetic_npz(str(tmp_path_factory.mktemp("data") / "synth.npz"), counts=(80, 30, 30))


@pytest.fixture(scope="session")
def trained_run(synth_npz, tmp_path_factory):
    from medxai.train import TrainConfig, train

    out = tmp_path_factory.mktemp("runs")
    cfg = TrainConfig(data_path=synth_npz, model="small_cnn", model_kwargs={"width": 8}, epochs=4,
                      batch_size=32, patience=10, out_dir=str(out), run_name="smoke", seed=0)
    return train(cfg, verbose=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
