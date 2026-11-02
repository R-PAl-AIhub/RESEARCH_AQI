"""Step 3.2 check: no feature at hour t uses data from after hour t."""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.build import build_features, feature_names

FEAT_CFG = {"lag_variables": ["PM2.5", "PM10"],
            "lags": [1, 3, 6, 12, 24], "rolling_hours": [3, 6, 12, 24], "rolling_stats": ["mean", "std"],
            "rolling_variables": ["PM2.5", "PM10"],
            "seasons": {m: "winter" if m in (12, 1, 2) else "other" for m in range(1, 13)}}
TARGETS = ["PM2.5", "PM10"]
CUT = pd.Timestamp("2024-01-03 12:00")
NO_LOOKAHEAD = ["deletion", "median", "ffill", "knn", "mice"]   # bfill copies later values back: tested apart


def _imp(method):
    """Imputation settings for the tests; median / knn / mice learn only from hours up to the cut."""
    return {"method": method, "train_end": str(CUT), "seed": 0, "knn_neighbors": 3, "knn_donors": 1000,
            "mice_iterations": 5}


def _fake_merged(seed=0):
    """A small merged-like table: 2 stations x 120 hours with short and long gaps.

    What: same column types as merged.parquet (a CPCB variable, WD, an era5_
          column, a flag column, row_present).
    How:  random values; one 3-hour gap and one 20-hour gap per station, the
          long one crossing the cut hour so the fill rule is tested there.
    """
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=120, freq="h")
    parts = []
    for station in ("A", "B"):
        g = pd.DataFrame({"station": station, "time": time, "row_present": True,
                          "PM2.5": rng.uniform(10, 200, 120), "PM10": rng.uniform(20, 300, 120),
                          "WD": rng.uniform(0, 360, 120), "AT": rng.uniform(15, 35, 120),
                          "era5_t2m": rng.uniform(15, 35, 120), "flag_range_AT": False})
        g.loc[20:22, ["PM2.5", "AT"]] = np.nan
        g.loc[52:71, ["PM2.5", "PM10", "WD"]] = np.nan     # hours 52-71 cross the cut (hour 60)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("method", NO_LOOKAHEAD)
def test_features_up_to_t_ignore_the_future(method):
    """Change every value after the cut hour; features up to the cut must not change (every method but bfill).

    What: builds features twice, once from the real table and once after all
          later values are replaced (new numbers, new gaps, filled old gaps).
    How:  if any feature at hour <= cut looked ahead, the two results would differ there.
    """
    merged = _fake_merged()
    changed = merged.copy()
    later = changed["time"] > CUT
    rng = np.random.default_rng(99)
    for col in ("PM2.5", "PM10", "WD", "AT", "era5_t2m"):
        new = rng.uniform(0, 500, later.sum())
        new[rng.random(later.sum()) < 0.3] = np.nan        # also move where the gaps are
        changed.loc[later, col] = new

    a, _ = build_features(merged, FEAT_CFG, TARGETS, _imp(method))
    b, _ = build_features(changed, FEAT_CFG, TARGETS, _imp(method))
    pdt.assert_frame_equal(a[a["time"] <= CUT].reset_index(drop=True),
                           b[b["time"] <= CUT].reset_index(drop=True))


def test_bfill_looks_ahead():
    """bfill fills a gap with the next later value, so hours before the cut DO change (the known leak).

    What: the same change as above; with bfill some feature at hour <= cut must differ.
    How:  documents why bfill is reported but never picked as the best method.
    """
    merged = _fake_merged()
    changed = merged.copy()
    later = changed["time"] > CUT
    changed.loc[later, "PM2.5"] = changed.loc[later, "PM2.5"] + 1000
    a, _ = build_features(merged, FEAT_CFG, TARGETS, _imp("bfill"))
    b, _ = build_features(changed, FEAT_CFG, TARGETS, _imp("bfill"))
    early = a["time"] <= CUT
    assert not a[early].reset_index(drop=True).equals(b[b["time"] <= CUT].reset_index(drop=True))


@pytest.mark.parametrize("method", ["deletion", "median", "ffill", "bfill", "knn", "mice"])
def test_targets_never_filled(method):
    """Every method leaves the target_ copies empty where PM was empty; deletion fills nothing, the others fill all.

    What: checks the long PM2.5 / PM10 gap of station A (hours 52-71).
    How:  target_pm25 / target_pm10 stay missing; the PM2.5 input is empty only under deletion.
    """
    features, _ = build_features(_fake_merged(), FEAT_CFG, TARGETS, _imp(method))
    a = features[features["station"] == "A"].reset_index(drop=True)
    assert a.loc[52:71, "target_pm25"].isna().all()
    assert a.loc[52:71, "target_pm10"].isna().all()
    assert "target_pm10" not in feature_names(features)
    if method == "deletion":
        assert a.loc[52:71, "PM2.5"].isna().all()
    else:
        assert a.loc[52:71, "PM2.5"].notna().all()
