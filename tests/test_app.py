import os

import numpy as np
import pytest
from PIL import Image


def test_gradio_app_builds(trained_run):
    pytest.importorskip("gradio")
    import app

    demo = app.build(os.path.join(trained_run["run_dir"], "best.pt"))
    assert demo is not None


def test_gradio_app_missing_weights(tmp_path):
    pytest.importorskip("gradio")
    import app

    assert app.build(str(tmp_path / "missing.pt")) is not None
