"""Part 4: train one forecaster (point or quantile), save its outputs, print the check.

Run one model per process:  python -m src.models.train --model gru --kind point
                            python -m src.models.train --model gru --kind quantile
"""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import load_config
from src.features.build import target_column
from src.models.forecasters import HEAD_LAYERS, KNOWN_FUTURE, build
from src.models.metrics import error_rows
from src.seed import set_seed

SPLIT_FILES = {"train": "train", "val": "validation", "calib": "calibration", "test": "test"}   # samples name -> file name


def short(target):
    """"PM2.5" -> "pm25", "PM10" -> "pm10" (used in column names)."""
    return target_column(target).removeprefix("target_")


def load_split(samples_dir, split):
    """Read one split's sample arrays from step 3.3.

    What: returns a dict with X (inputs), y (scaled targets), y_raw (µg/m³
          targets), both (samples, targets, horizons), meta (station,
          timestamp, season, ...), and the two extra TFT inputs: K (known future
          calendar, (samples, horizons, 4)) and S (station index, (samples, 1)).
    How:  np.load for the .npy arrays, pandas for meta_<split>.csv (CSV only in
          PROJECT_2; timestamp read as a date); horizons from target_layout.json,
          the fixed station order from stations.json (both step 3.3).
    """
    meta = pd.read_csv(samples_dir / f"meta_{split}.csv", parse_dates=["timestamp"])
    horizons = json.loads((samples_dir / "target_layout.json").read_text(encoding="utf-8"))["horizons"]
    stations = json.loads((samples_dir / "stations.json").read_text(encoding="utf-8"))
    return {"X": np.load(samples_dir / f"X_{split}.npy"),
            "y": np.load(samples_dir / f"y_{split}.npy"),
            "y_raw": np.load(samples_dir / f"y_raw_{split}.npy"),
            "meta": meta,
            "K": known_future(meta, horizons),
            "S": meta["station"].map({s: i for i, s in enumerate(stations)}).to_numpy("int32")[:, None]}


def known_future(meta, horizons):
    """Calendar values known in advance for every horizon (the TFT future input).

    What: (samples, horizons, 4): sin and cos of the hour of day and of the day of
          year at timestamp + h (order KNOWN_FUTURE).
    How:  the same formulas as step 3.2 add_cyclic, on the plain calendar (values
          -1..1, not scaled); the timestamp is the last input hour.
    """
    times = pd.DatetimeIndex((meta["timestamp"].to_numpy()[:, None]
                              + np.array(horizons, dtype="timedelta64[h]")).ravel())
    hour, doy = 2 * np.pi * times.hour / 24, 2 * np.pi * (times.dayofyear - 1) / 365.25
    values = np.stack([np.sin(hour), np.cos(hour), np.sin(doy), np.cos(doy)], axis=1)
    return values.reshape(len(meta), len(horizons), len(KNOWN_FUTURE)).astype("float32")


def model_inputs(model, split, rows=slice(None)):
    """The arrays a model reads: X for most models, [X, K, S] for the TFT (three inputs).

    What: returns X[rows], or the list [X[rows], K[rows], S[rows]].
    How:  looks at how many inputs the Keras model has.
    """
    if len(model.inputs) == 3:
        return [split["X"][rows], split["K"][rows], split["S"][rows]]
    return split["X"][rows]


def to_ugm3(y_scaled, scaler, targets):
    """Turn scaled log values back into µg/m³, for every target.

    What: y_scaled (samples, targets, horizons) -> same shape in µg/m³, never below 0.
    How:  undoes step 3.3 per target: multiply by that target column's train
          spread, add its train mean (both from the saved scaler), then exp(v) - 1.
    """
    names = list(scaler.feature_names_in_)
    out = np.empty_like(y_scaled, dtype=np.float64)
    for t, target in enumerate(targets):
        i = names.index(target)
        out[:, t] = np.clip(np.expm1(y_scaled[:, t] * scaler.scale_[i] + scaler.mean_[i]), 0, None)
    return out


def heads_read_embedding_only(model):
    """True if every output head's only input is the "embedding" layer.

    What: checks the rule Worker B relies on (no skip connection around the embedding).
    How:  compares the input tensor of each head layer with the embedding output.
    """
    emb = model.get_layer("embedding").output
    heads = [layer for layer in model.layers if layer.name in HEAD_LAYERS]
    return bool(heads) and all(layer.input is emb for layer in heads)


def model_tag(model):
    """Version tag = model name + first 10 characters of a hash of all weights.

    What: e.g. "gru_quantile_3f9a1c07be". Same weights -> same tag; any retrain
          -> a new tag, so Worker B and C never mix outputs of two models.
    How:  sha256 over the bytes of every weight array, in layer order.
    """
    digest = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
    return f"{model.name}_{digest[:10]}"


def split_lines(pred, n_targets, n_horizons):
    """Cut the quantile model output into low, median and high arrays.

    What: returns three arrays of shape (samples, targets, horizons).
    How:  the output order is [low, median, high], each targets x horizons long.
    """
    lines = pred.reshape(-1, 3, n_targets, n_horizons)
    return lines[:, 0], lines[:, 1], lines[:, 2]


def output_table(split, model, embed_model, scaler, targets, horizons):
    """One row per sample: who/when, real target, median/low/high in µg/m³, embedding.

    What: columns station, timestamp, season, then per target t and horizon h:
          target_<t>_h<h>, median_<t>_h<h>, low_<t>_h<h>, high_<t>_h<h>; then emb_1 .. emb_d.
    How:  predicts the split, undoes the scaling and the log for the three
          lines, and takes the untouched µg/m³ target from step 3.3.
    """
    pred = model.predict(model_inputs(model, split), batch_size=1024, verbose=0)
    low, median, high = (to_ugm3(a, scaler, targets) for a in split_lines(pred, len(targets), len(horizons)))
    embedding = embed_model.predict(model_inputs(embed_model, split), batch_size=1024, verbose=0)
    table = split["meta"][["station", "timestamp", "season"]].copy()
    for t, target in enumerate(targets):
        for i, h in enumerate(horizons):
            key = f"{short(target)}_h{h}"
            table[f"target_{key}"] = split["y_raw"][:, t, i]
            table[f"median_{key}"], table[f"low_{key}"], table[f"high_{key}"] = \
                median[:, t, i], low[:, t, i], high[:, t, i]
    for j in range(embedding.shape[1]):
        table[f"emb_{j + 1}"] = embedding[:, j]
    return table


def load(model_name, split):
    """Read one split of a quantile forecaster's outputs. Later steps use this.

    What: model_name = gru, lstm, cnn, cnn_lstm or tft; split = "train",
          "calibration" or "test"; returns the table written by output_table.
    How:  reads forecasts_dir/<model>/quantile/<split>.csv; paths come from config.yaml.
    """
    cfg = load_config()
    return pd.read_csv(cfg["forecasts_dir"] / model_name / "quantile" / f"{split}.csv", parse_dates=["timestamp"])


def coverage(table, targets, horizons):
    """Share of real targets between the low and high line, per target and horizon.

    What: a Series indexed by (target, horizon), e.g. 0.90 = 90% inside the band.
    How:  counts low <= target <= high and divides by the number of samples.
    """
    out = {}
    for target in targets:
        for h in horizons:
            k = f"{short(target)}_h{h}"
            out[(target, h)] = ((table[f"low_{k}"] <= table[f"target_{k}"]) &
                                (table[f"target_{k}"] <= table[f"high_{k}"])).mean()
    return pd.Series(out)


def lines_array(table, kind, targets, horizons):
    """Pick the <kind>_<t>_h<h> columns of an output table as (samples, targets, horizons)."""
    return np.stack([table[[f"{kind}_{short(t)}_h{h}" for h in horizons]].to_numpy() for t in targets], axis=1)


def fit(model, train, val, cfg):
    """Train on all train samples (2023-24) with early stopping on the validation split (Jan-Apr 2025).

    What: returns the history table. The best validation epoch's weights are kept.
    How:  shared settings from config.yaml model (same for every forecaster).
          Step 3.3 already keeps every sample inside one split, so no buffer is needed.
    """
    m_cfg = cfg["model"]
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=m_cfg["patience"], restore_best_weights=True)
    history = model.fit(model_inputs(model, train), train["y"],
                        validation_data=(model_inputs(model, val), val["y"]),
                        epochs=m_cfg["max_epochs"], batch_size=m_cfg["batch_size"], callbacks=[stop], verbose=2)
    return pd.DataFrame(history.history).rename_axis("epoch")


def run_point(cfg, name, model, embed_model, train, history, out_dir):
    """Point model: predict calibration, save it, return the check lines.

    What: saves calib_pred.csv (station, timestamp, pred_<t>_h<h> in µg/m³) and
          returns lines with the calibration RMSE of this model.
    How:  every calibration sample is scored, as in compare.py.
    """
    targets, horizons, samples_dir = cfg["targets"], cfg["horizons"], cfg["samples_dir"]
    scaler = joblib.load(samples_dir / "scaler.joblib")
    calib = load_split(samples_dir, "calib")
    pred = to_ugm3(model.predict(model_inputs(model, calib), batch_size=1024, verbose=0), scaler, targets)
    table = calib["meta"][["station", "timestamp"]].copy()
    for t, target in enumerate(targets):
        for i, h in enumerate(horizons):
            table[f"pred_{short(target)}_h{h}"] = pred[:, t, i]
    table.to_csv(out_dir / "calib_pred.csv", index=False)

    errors = error_rows(calib["y_raw"], {name: pred}, targets, horizons)
    embedding = embed_model.predict(model_inputs(embed_model, calib, slice(0, 2048)), batch_size=1024, verbose=0)
    return [f"\ncalibration RMSE in µg/m³ ({len(pred)} samples):",
            errors.pivot_table(index=["target", "horizon"], columns="model", values="rmse").to_string(),
            f"embedding shape (first 2048 calibration samples): {embedding.shape}"]


def run_quantile(cfg, model, embed_model, train, val, out_dir):
    """Quantile model: export every split, check coverage, reload, return the check lines.

    What: saves train/validation/calibration/test .csv (output_table) and tag.txt; returns
          lines with train and calibration coverage, calibration median RMSE,
          low-above-high count and the reload check. Nothing about test is printed.
    How:  reload check = coverage recomputed from load() must match, and the
          reloaded model.keras must give the same predictions.
    """
    targets, horizons, samples_dir = cfg["targets"], cfg["horizons"], cfg["samples_dir"]
    scaler = joblib.load(samples_dir / "scaler.joblib")
    tables = {}
    for split, file in SPLIT_FILES.items():
        data = train if split == "train" else val if split == "val" else load_split(samples_dir, split)
        tables[file] = output_table(data, model, embed_model, scaler, targets, horizons)
        tables[file].to_csv(out_dir / f"{file}.csv", index=False)
        if split == "calib":
            calib_in = model_inputs(model, data, slice(0, 4096))
        if split in ("calib", "test"):
            del data
    tag = model_tag(model)
    (out_dir / "tag.txt").write_text(tag, encoding="utf-8")

    cover = pd.DataFrame({f: coverage(tables[f], targets, horizons) for f in ("train", "calibration")}).round(3)
    truth, med = (lines_array(tables["calibration"], k, targets, horizons) for k in ("target", "median"))
    med_rmse = error_rows(truth, {"median": med}, targets, horizons).set_index(["target", "horizon"])["rmse"]
    low_above_high = sum(int((tables[f][f"low_{short(t)}_h{h}"] > tables[f][f"high_{short(t)}_h{h}"]).sum())
                         for f in ("train", "calibration") for t in targets for h in horizons)
    model_name = model.name.removesuffix("_quantile")
    same = np.allclose(coverage(load(model_name, "train"), targets, horizons), cover["train"], atol=1e-3)
    again = keras.models.load_model(out_dir / "model.keras", compile=False)
    diff = float(np.abs(again.predict(calib_in, verbose=0) - model.predict(calib_in, verbose=0)).max())
    return [f"model tag: {tag}",
            f"\nshare of targets between low and high (aim {1 - cfg['alpha']:.2f}):", cover.to_string(),
            "\ncalibration RMSE of the median line (µg/m³):", med_rmse.to_string(),
            f"\nlow above high (train + calibration): {low_above_high}",
            f"reloaded csv reproduces coverage: {same}; reloaded model.keras max difference: {diff:.2e}",
            f"embedding columns: {[c for c in tables['train'].columns if c.startswith('emb_')]}"]


def main():
    """Train one forecaster, save model + history + outputs, print and save the check.

    What: --model gru|lstm|cnn|cnn_lstm|tft, --kind point|quantile. Saves to
          forecasts_dir/<model>/<kind>/: model.keras, history.csv, report.txt and
          calib_pred.csv (point) or train/calibration/test.csv + tag.txt (quantile).
    How:  set_seed first; all sizes, dates and the d of the embedding come from
          config.yaml; calibration and test are only predicted, never trained on.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["point", "quantile"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    samples_dir = cfg["samples_dir"]
    train = load_split(samples_dir, "train")
    n_features = len(json.loads((samples_dir / "feature_names.json").read_text(encoding="utf-8")))
    _, n_hours, _ = train["X"].shape
    n_targets, n_horizons = train["y"].shape[1:]
    alpha = cfg["alpha"] if args.kind == "quantile" else None
    n_stations = len(json.loads((samples_dir / "stations.json").read_text(encoding="utf-8")))
    model, embed_model = build(args.model, n_hours, n_features, n_targets, n_horizons, cfg["d"],
                               cfg["forecasters"], cfg["model"], alpha, n_stations)

    out_dir = cfg["forecasts_dir"] / args.model / args.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    val = load_split(samples_dir, "val")
    history = fit(model, train, val, cfg)
    history.to_csv(out_dir / "history.csv")
    model.save(out_dir / "model.keras")

    val_ts = val["meta"]["timestamp"]
    lines = [f"{model.name}: {model.count_params()} weights; outputs read the embedding only: "
             f"{heads_read_embedding_only(model)}; embedding size d = {cfg['d']}",
             f"train samples: {len(train['X'])}  validation samples: {len(val['X'])}  "
             f"dates {val_ts.min()} to {val_ts.max()} (validation split)",
             f"epochs run: {len(history)}  best val_loss: {history['val_loss'].min():.4f} "
             f"at epoch {history['val_loss'].idxmin() + 1}"]
    if args.kind == "point":
        lines += run_point(cfg, args.model, model, embed_model, train, history, out_dir)
    else:
        lines += run_quantile(cfg, model, embed_model, train, val, out_dir)
    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
