"""Step 3.3 checks: targets come after inputs, no sample crosses a split, the scaler uses TRAIN only."""
import numpy as np
import pandas as pd

from src.features.samples import make_samples, split_labels

SPLITS = {"train_start": "2024-01-01 00:00", "train_end": "2024-01-04 23:00",
          "val_start": "2024-01-05 00:00", "val_end": "2024-01-06 23:00",
          "calib_start": "2024-01-07 00:00", "calib_end": "2024-01-08 23:00",
          "test_start": "2024-01-09 00:00", "test_end": "2024-01-10 23:00"}
LOG_VARIABLES = ["PM2.5", "PM10", "NO"]
INPUT_HOURS, HORIZONS, MAX_MISSING = 6, [1, 3], 0.10
TARGETS = {"PM2.5": "target_pm25", "PM10": "target_pm10"}


def _fake_features(seed=0):
    """A small features-like table: 2 stations x 10 days, hourly, with a few gaps.

    What: same column roles as features.csv (concentrations, a lag, weather,
          WD as sine/cosine, target_pm25, target_pm10, season).
    How:  random values; a few missing inputs and targets (of each target) so
          samples get dropped.
    """
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=240, freq="h")
    parts = []
    for station in ("A", "B"):
        pm, pm10 = rng.uniform(5, 300, 240), rng.uniform(10, 400, 240)
        g = pd.DataFrame({"station": station, "time": time, "row_present": True,
                          "target_pm25": pm, "target_pm10": pm10, "PM2.5": pm, "PM10": pm10,
                          "NO": rng.uniform(-2, 80, 240), "AT": rng.uniform(15, 35, 240),
                          "WD_sin": rng.uniform(-1, 1, 240), "WD_cos": rng.uniform(-1, 1, 240),
                          "season": "DJF"})
        g["PM2.5_lag1"] = g["PM2.5"].shift(1)
        g.loc[30:33, "target_pm25"] = np.nan
        g.loc[150:151, "target_pm10"] = np.nan
        g.loc[100:101, ["NO", "AT"]] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _run(features):
    """Run the step on a table with the small test settings."""
    return make_samples(features, SPLITS, list(TARGETS), LOG_VARIABLES, INPUT_HOURS, HORIZONS, MAX_MISSING)


def test_targets_come_after_inputs():
    """Each target (PM2.5 and PM10) is its real value at (last input hour + h), with h > 0.

    What: for every sample, looks up target_pm25 and target_pm10 at timestamp + h
          in the source table and compares them with y_raw[sample, target, horizon].
    How:  if a target were taken from an input hour or earlier, or the target axis
          were mixed up, the values or the hours would not match.
    """
    features = _fake_features()
    samples, *_ = _run(features)
    source = features.set_index(["station", "time"])
    assert all(h > 0 for h in HORIZONS)
    for s in samples.values():
        assert s["y_raw"].shape[1:] == (len(TARGETS), len(HORIZONS))
        for i, row in s["meta"].iterrows():
            for t, col in enumerate(TARGETS.values()):
                for j, h in enumerate(HORIZONS):
                    expected = source.loc[(row["station"], row["timestamp"] + pd.Timedelta(hours=h)), col]
                    assert np.isclose(s["y_raw"][i, t, j], expected, rtol=1e-5)


def test_no_sample_crosses_a_split():
    """The first input hour and the last target hour are in the sample's own split.

    What: checks every kept sample of every split.
    How:  recomputes the split of (timestamp - input_hours + 1) and of
          (timestamp + largest horizon) with the same date rules.
    """
    samples, *_ = _run(_fake_features())
    for name, s in samples.items():
        meta = s["meta"]
        assert len(meta) > 0
        first = split_labels(meta["timestamp"] - pd.Timedelta(hours=INPUT_HOURS - 1), SPLITS)
        last = split_labels(meta["timestamp"] + pd.Timedelta(hours=max(HORIZONS)), SPLITS)
        assert (first == name).all() and (last == name).all()


def test_scaler_uses_train_dates_only():
    """Changing every non-TRAIN value must not change the scaler.

    What: builds twice: once as is, once with all calibration/test values
          multiplied by 10 and shifted by 50.
    How:  if the scaler had seen any non-TRAIN row, its mean or spread would move.
    """
    features = _fake_features()
    changed = features.copy()
    later = changed["time"] > pd.Timestamp(SPLITS["train_end"])
    cols = ["PM2.5", "PM10", "NO", "AT", "WD_sin", "WD_cos", "PM2.5_lag1"]
    changed.loc[later, cols] = changed.loc[later, cols] * 10 + 50

    _, scaler_a, *_ = _run(features)
    _, scaler_b, *_ = _run(changed)
    np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_)
