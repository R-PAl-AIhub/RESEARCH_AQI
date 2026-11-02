"""Step 3.3: log + scale the features (scaler fitted on TRAIN only) and cut them into model samples."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.features.build import feature_names, target_column
from src.plot_style import table_figure
from src.temp_out import save_table, save_text

SPLITS = ("train", "val", "calib", "test")    # option C: train 2023-24; 2025 = val Jan-Apr, calib May-Aug, test Sep-Dec


def split_labels(times, splits):
    """Which split each hour belongs to: "train", "val", "calib", "test" or None.

    What: returns one label per row.
    How:  compares each time with the start and end dates in config.yaml
          (both ends included). Hours outside every split get None.
    """
    labels = pd.Series(None, index=times.index, dtype=object)
    for name in SPLITS:
        start, end = pd.Timestamp(splits[f"{name}_start"]), pd.Timestamp(splits[f"{name}_end"])
        labels[(times >= start) & (times <= end)] = name
    return labels


def group_columns(names, bases):
    """Columns that belong to a list of base variables.

    What: e.g. bases ["PM2.5"] -> PM2.5, PM2.5_lag1, ..., PM2.5_mean24h;
          bases ["WD"] -> WD_sin, WD_cos.
    How:  a column belongs if it equals a base name or starts with "<base>_".
          ("NO_" does not match "NO2", so the gases stay apart.)
    """
    return [c for c in names if any(c == b or c.startswith(f"{b}_") for b in bases)]


def prepare_table(features, feature_cols, conc_cols, train_mask):
    """Log-transform the concentrations, then scale every feature with a TRAIN-only scaler.

    What: returns (scaled table, fitted scaler). Missing values stay missing.
    How:  values below 0 become 0 (log of a negative fails), then log(1 + x)
          shrinks the rare huge values. StandardScaler learns each column's mean
          and spread from the TRAIN rows only (it skips missing values) and then
          turns every row into (value - mean) / spread.
    """
    table = features.copy()
    table[conc_cols] = np.log1p(table[conc_cols].clip(lower=0))
    scaler = StandardScaler().fit(table.loc[train_mask, feature_cols])
    table[feature_cols] = scaler.transform(table[feature_cols])
    return table, scaler


def scale_target(raw, scaler, target):
    """Transform one µg/m³ target exactly like its own input column (PM2.5 or PM10).

    What: returns (log(1 + max(raw, 0)) - mean) / spread, with that column's
          train mean and spread from the scaler.
    How:  looks up the target column's position in the scaler; missing stays missing.
    """
    i = list(scaler.feature_names_in_).index(target)
    return (np.log1p(raw.clip(lower=0)) - scaler.mean_[i]) / scaler.scale_[i]


def station_samples(g, feature_cols, target_cols, input_hours, horizons, max_missing):
    """Cut one station's hourly table into samples.

    What: returns (X, y, y_raw, meta) for every kept sample. X = last
          `input_hours` hours of scaled features;
          y / y_raw = every target at each horizon after the last input hour
          (scaled / µg/m³), shape (samples, targets, horizons); meta = station,
          timestamp (last input hour), season, split.
    How:  rows are consecutive hours, so sample e uses rows e-47..e as input and
          rows e+h as targets. A sample is kept only if (1) its first input hour
          and its last target hour are in the same split (splits are blocks of
          time, so everything between is too), (2) no target (PM2.5 or PM10) is
          missing at any horizon and (3) at most `max_missing` of its input cells
          are missing. Remaining missing inputs become 0 (= the train average
          after scaling).
    """
    feats = g[feature_cols].to_numpy(np.float32)
    labels = g["split"].to_numpy()
    y_raw = g[target_cols].to_numpy()                                   # (rows, targets)
    y_scaled = g[[f"{c}_scaled" for c in target_cols]].to_numpy()

    last_h = max(horizons)
    ends = np.arange(input_hours - 1, len(g) - last_h)
    starts = ends - input_hours + 1
    target_rows = ends[:, None] + np.array(horizons)

    same_split = pd.notna(labels[starts]) & (labels[starts] == labels[ends + last_h])
    targets_ok = ~np.isnan(y_raw[target_rows]).any(axis=(1, 2))
    missing_per_row = np.concatenate([[0], np.cumsum(np.isnan(feats).sum(axis=1))])
    missing_share = (missing_per_row[ends + 1] - missing_per_row[starts]) / (input_hours * feats.shape[1])
    keep = same_split & targets_ok & (missing_share <= max_missing)
    e = ends[keep]

    windows = np.lib.stride_tricks.sliding_window_view(feats, input_hours, axis=0)   # (rows, columns, hours)
    X = np.nan_to_num(windows[e - input_hours + 1].transpose(0, 2, 1), nan=0.0).astype(np.float32)
    rows = e[:, None] + np.array(horizons)
    meta = pd.DataFrame({"station": g["station"].to_numpy()[e], "timestamp": g["time"].to_numpy()[e],
                         "season": g["season"].to_numpy()[e], "split": labels[e]})
    # rows-of-targets come out as (samples, horizons, targets); store as (samples, targets, horizons)
    return (X, y_scaled[rows].transpose(0, 2, 1).astype(np.float32),
            y_raw[rows].transpose(0, 2, 1).astype(np.float32), meta)


def make_samples(features, splits, targets, log_variables, input_hours, horizons, max_missing):
    """The whole step on a features table (used by main and by the tests).

    What: returns (samples, scaler, names, table). samples[split] holds X, y,
          y_raw (both (samples, targets, horizons)) and meta; names = feature
          names in array order; table = the scaled hourly table (for the
          mean/spread check).
    How:  label each hour with its split, log + scale with a TRAIN-only scaler,
          transform each target like its own input column, cut each station into
          samples, then stack the samples of each split.
    """
    df = features.sort_values(["station", "time"]).reset_index(drop=True)
    df["split"] = split_labels(df["time"], splits)
    feature_cols = feature_names(features)
    conc_cols = group_columns(feature_cols, log_variables)
    table, scaler = prepare_table(df, feature_cols, conc_cols, df["split"] == "train")
    target_cols = [target_column(t) for t in targets]
    for t, col in zip(targets, target_cols):
        table[f"{col}_scaled"] = scale_target(df[col], scaler, t)

    parts = [station_samples(g, feature_cols, target_cols, input_hours, horizons, max_missing)
             for _, g in table.groupby("station")]
    samples = {}
    for name in SPLITS:
        masks = [m["split"].to_numpy() == name for *_, m in parts]
        picks = [(X[k], y[k], yr[k], m[k]) for (X, y, yr, m), k in zip(parts, masks)]
        samples[name] = {"X": np.concatenate([p[0] for p in picks]),
                         "y": np.concatenate([p[1] for p in picks]),
                         "y_raw": np.concatenate([p[2] for p in picks]),
                         "meta": pd.concat([p[3] for p in picks], ignore_index=True)}
    return samples, scaler, feature_cols, table


def deletion_hours(features, feature_cols):
    """Hours the deletion rule cannot use, per station and year, plus one total row.

    What: table indexed by (station, year): hours (all hours of that station-year on the
          step 2.2 grid), deleted (hours with at least one empty input cell, after step
          3.2), deleted % (deleted / hours x 100); last row "all stations · all years".
    How:  an hour with any empty input cell can never sit inside a complete 48-hour
          window, so with deletion every window that covers it is dropped.
    """
    frame = pd.DataFrame({"station": features["station"].str.replace(", Kolkata - WBPCB", ""),
                          "year": features["time"].dt.year,
                          "empty": features[feature_cols].isna().any(axis=1)})
    table = frame.groupby(["station", "year"])["empty"].agg(hours="size", deleted="sum")
    total = pd.DataFrame({"hours": [table["hours"].sum()], "deleted": [table["deleted"].sum()]},
                         index=pd.MultiIndex.from_tuples([("all stations", "all years")], names=["station", "year"]))
    table = pd.concat([table, total])
    table["deleted %"] = (table["deleted"] / table["hours"] * 100).round(1)
    return table


def spread_check(table, feature_cols):
    """Mean and spread of every scaled feature, per split.

    What: one row per feature, columns <split>_mean and <split>_std.
    How:  computed on the hourly table (missing values skipped). TRAIN must be
          about 0 and 1, because the scaler learned from it; the other splits
          only come close.
    """
    out = {}
    for name in SPLITS:
        part = table.loc[table["split"] == name, feature_cols]
        out[f"{name}_mean"], out[f"{name}_std"] = part.mean(), part.std()
    return pd.DataFrame(out).round(3)


def main():
    """Run step 3.3: save the scaler, the sample arrays and the feature names, print the check.

    What: artifacts/samples/ gets scaler.joblib, feature_names.json,
          target_layout.json (axis order and names of y) and per split
          X_<split>.npy, y_<split>.npy, y_raw_<split>.npy (float32; y is
          (samples, targets, horizons)) plus meta_<split>.csv (station, timestamp,
          season, µg/m³ targets; CSV only). processed_dir gets model_table.csv:
          station, time and every model input column per hour, exactly as X
          holds them (log + scaled, empty -> 0). Prints shapes,
          kept samples per split and station, and mean/spread.
    How:  reads features.csv (never changes it); targets, dates, sizes and groups
          come from config.yaml.
    """
    cfg = load_config()
    s_cfg, horizons, targets = cfg["samples"], cfg["horizons"], cfg["targets"]
    features = pd.read_csv(cfg["processed_dir"] / cfg["features_file"], parse_dates=["time"])
    method = cfg["imputation"]["method"]
    max_missing = 0.0 if method == "deletion" else s_cfg["max_missing_input"]   # deletion: no empty cell at all
    samples, scaler, names, table = make_samples(features, cfg["splits"], targets, s_cfg["log_variables"],
                                                 s_cfg["input_hours"], horizons, max_missing)

    out_dir = cfg["samples_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.joblib")
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    layout = {"y_axes": ["sample", "target", "horizon"], "targets": targets, "horizons": horizons}
    (out_dir / "target_layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    stations = sorted(features["station"].unique())          # one fixed order for the TFT station input
    (out_dir / "stations.json").write_text(json.dumps(stations, indent=1), encoding="utf-8")
    for name, s in samples.items():
        for key in ("X", "y", "y_raw"):
            np.save(out_dir / f"{key}_{name}.npy", s[key])
        meta = s["meta"].assign(**{f"{target_column(t).removeprefix('target_')}_ugm3_h{h}": s["y_raw"][:, ti, hi]
                                   for ti, t in enumerate(targets) for hi, h in enumerate(horizons)})
        meta.to_csv(out_dir / f"meta_{name}.csv", index=False)   # readable in Excel

    view = table[["station", "time"] + names].fillna(0)      # exactly what X holds per hour (empty -> 0)
    view.to_csv(cfg["processed_dir"] / cfg["model_table_file"], index=False)

    all_meta = pd.concat([s["meta"] for s in samples.values()], ignore_index=True)
    kept = pd.crosstab(all_meta["station"], all_meta["split"])
    kept = kept.reindex(columns=list(SPLITS))
    spread = spread_check(table, names)
    if method == "deletion":                                  # which hours the deletion rule throws away
        gone = deletion_hours(features, names)
        cfg["figures_dir"].mkdir(parents=True, exist_ok=True)
        gone.to_csv(cfg["figures_dir"] / cfg["imputation"]["deletion_file"])
        table_figure(gone, "Deletion: hours with at least one empty input cell, per station and year",
                     cfg["figures_dir"] / cfg["imputation"]["deletion_plot_file"], pct_cols=("deleted %",))
    lines = [f"imputation method: {method}; window dropped if more than {max_missing:.0%} of its input cells are empty",
             f"{len(names)} features per hour (no miss_ flags), {s_cfg['input_hours']} hours per sample, "
             f"horizons {horizons}",
             f"targets {targets}; y axes (sample, target, horizon)",
             "\narray shapes:"]
    lines += [f"  {n:5s} X {s['X'].shape}  y {s['y'].shape}  y_raw {s['y_raw'].shape}" for n, s in samples.items()]
    lines += ["\nsamples kept per station and split:", kept.to_string(),
              "\nscaled features, range over all features of the mean and spread per split:"]
    for n in SPLITS:
        m, sd = spread[f"{n}_mean"], spread[f"{n}_std"]
        lines.append(f"  {n:5s} mean {m.min():+.3f} to {m.max():+.3f}   spread {sd.min():.3f} to {sd.max():.3f}")
    lines.append(f"\nsaved to: {out_dir}")
    text = "\n".join(lines)
    print(text)

    save_text(cfg, "3.3_samples_report.txt", text)
    save_table(cfg, "3.3_kept_per_station.csv", kept)
    save_table(cfg, "3.3_feature_mean_spread.csv", spread)
    print(f"temp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
