"""Part 4 checks: every forecaster reads its outputs from the embedding only and keeps low <= median <= high."""
import numpy as np
import pytest

from src.models.forecasters import build
from src.models.train import heads_read_embedding_only, split_lines

F_CFG = {"gru": {"units": [8, 8]}, "lstm": {"units": [8, 8]}, "cnn": {"filters": [8, 8], "kernel_size": 3},
         "cnn_lstm": {"filters": [8], "kernel_size": 3, "lstm_units": [8]}, "tft": {"hidden": 8, "heads": 2}}
M_CFG = {"dropout": 0.2, "learning_rate": 0.001}
HOURS, FEATURES, TARGETS, HORIZONS, D = 12, 5, 2, 4, 6


def _inputs(model, x):
    """X for most models; [X, known future calendar, station index] for the TFT."""
    if len(model.inputs) == 3:
        n = len(x)
        return [x, np.zeros((n, HORIZONS, 4), "float32"), np.zeros((n, 1), "int32")]
    return x


@pytest.mark.parametrize("name", list(F_CFG))
def test_heads_read_embedding_only(name):
    """Point and quantile versions: heads read only the embedding; shapes are right.

    What: builds each model small and checks the head inputs and output shapes.
    How:  heads_read_embedding_only compares each head's input with the embedding tensor.
    """
    x = np.random.default_rng(0).normal(size=(3, HOURS, FEATURES)).astype("float32")
    for alpha in (None, 0.10):
        model, embed = build(name, HOURS, FEATURES, TARGETS, HORIZONS, D, F_CFG, M_CFG, alpha)
        assert heads_read_embedding_only(model)
        assert embed(_inputs(model, x)).shape == (3, D)
        n = 1 if alpha is None else 3
        assert np.prod(model(_inputs(model, x)).shape[1:]) == n * TARGETS * HORIZONS


@pytest.mark.parametrize("name", list(F_CFG))
def test_low_never_above_high(name):
    """An untrained quantile model already keeps low <= median <= high.

    What: random inputs, random weights.
    How:  the gaps are softplus outputs, which are never negative.
    """
    x = np.random.default_rng(1).normal(size=(16, HOURS, FEATURES)).astype("float32")
    model, _ = build(name, HOURS, FEATURES, TARGETS, HORIZONS, D, F_CFG, M_CFG, 0.10)
    low, median, high = split_lines(np.asarray(model(_inputs(model, x))), TARGETS, HORIZONS)
    assert (low <= median).all() and (median <= high).all()
