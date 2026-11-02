"""Teacher update (2026-11-02): SHAP explanation of the published forecast, every imputation run x model
(explanation only, chooses nothing).

Model = the median line of each quantile forecaster, wrapped so its outputs are µg/m³ (the same back-transform
as step 4: undo the scaling, then exp(v) - 1). SHAP by expected gradients against random train windows, on
random calibration windows: shap.GradientExplainer for GRU, LSTM, 1D CNN and CNN + LSTM; for the TFT the same
formula computed directly (its station input is a whole number, which shap cannot take a gradient of), with the
window's own calendar and station kept fixed - only the 48-hour input window is explained. The direct code is
checked against the shap library on one pair (shap.check_run / check_model).
Each input's SHAP value is summed over its 48 input hours, so one number per input and window.
"""
import json
import os

import joblib
import keras
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

import src.models.forecasters  # noqa: F401  (registers the TFT's custom layers so its model.keras loads)
from src.config import load_config
from src.eval.imputation_compare import run_config
from src.models.compare import INK, LABELS, MUTED, SURFACE, style
from src.models.train import known_future, short
from src.temp_out import save_text

SECONDARY, RULE = "#52514e", "#c3c2b7"
HEAT = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
VALUE_CMAP = LinearSegmentedColormap.from_list("low_high", ["#2a78d6", "#c3c2b7", "#e34948"])   # low - mid - high
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#2a78d6", "#f0efec", "#e34948"])       # -1 - 0 - +1


def save(fig, path):
    """Save as drawn: no grid, and NO text rewriting, so the input names stay exactly as in feature_names.json."""
    for ax in fig.axes:
        ax.grid(False)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def wrapped_model(cfg, rc, model_name):
    """The quantile forecaster cut to its median outputs for the explained targets x horizons, in µg/m³.

    What: returns (keras model, list of (target, horizon) per output column).
    How:  output order of the forecaster is [low, median, high] x targets x horizons (train.split_lines);
          keeps the median block, then value * scale + mean of the target's scaler column, then exp(v) - 1.
    """
    base = keras.models.load_model(rc["forecasts_dir"] / model_name / "quantile" / "model.keras", compile=False)
    scaler = joblib.load(rc["samples_dir"] / "scaler.joblib")
    names, targets, horizons = list(scaler.feature_names_in_), cfg["targets"], cfg["horizons"]
    outs = [(t, h) for t in targets for h in cfg["shap"]["horizons"]]
    idx = [len(targets) * len(horizons) + targets.index(t) * len(horizons) + horizons.index(h) for t, h in outs]
    mean = np.array([scaler.mean_[names.index(t)] for t, _ in outs], "float32")
    scale = np.array([scaler.scale_[names.index(t)] for t, _ in outs], "float32")
    to_ugm3 = keras.layers.Lambda(lambda v: keras.ops.expm1(keras.ops.take(v, idx, axis=1) * scale + mean),
                                  name="median_ugm3")             # a layer: shap counts outputs on the last layer
    return keras.Model(base.inputs, to_ugm3(base.outputs[0])), outs


def windows(rc, split, n, seed):
    """n random windows of one split: (X, calendar K, station S, row numbers), X read without loading the whole array.

    K and S are the TFT's extra inputs (train.load_split builds them the same way); other models use X only.
    """
    x = np.load(rc["samples_dir"] / f"X_{split}.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(seed).choice(len(x), size=min(n, len(x)), replace=False))
    meta = pd.read_csv(rc["samples_dir"] / f"meta_{split}.csv", parse_dates=["timestamp"]).iloc[rows]
    horizons = json.loads((rc["samples_dir"] / "target_layout.json").read_text(encoding="utf-8"))["horizons"]
    stations = json.loads((rc["samples_dir"] / "stations.json").read_text(encoding="utf-8"))
    s = meta["station"].map({st: i for i, st in enumerate(stations)}).to_numpy("int32")[:, None]
    return np.asarray(x[rows], dtype="float32"), known_future(meta, horizons), s, rows


def direct_expected_gradients(model, x, k, s, background, nsamples, batch, seed):
    """SHAP values of the 48-hour window by expected gradients, computed directly (used for the TFT).

    What: returns (outputs, windows, 48, features). For each window: average over nsamples draws of
          (x - b) * gradient of the output at b + a (x - b), b = a random background window, a ~ U(0, 1);
          the window's own calendar k and station s stay fixed (the same formula shap.GradientExplainer uses).
    How:  batched tf.GradientTape; one gradient pass per output on the summed batch output (each row of the
          output depends only on its own row of input, so this gives every row's own gradient). A fresh tape per
          output: a persistent tape gave NaN gradients for the TFT's later outputs (fresh tapes give none).
    """
    rng = np.random.default_rng(seed)
    n_out = model.outputs[0].shape[-1]
    out = np.zeros((n_out,) + x.shape, dtype="float32")
    rows_per = max(1, batch // nsamples)
    for i0 in range(0, len(x), rows_per):
        xi = x[i0:i0 + rows_per]
        idx = np.repeat(np.arange(len(xi)), nsamples)
        b = background[rng.integers(0, len(background), len(idx))]
        a = rng.uniform(0, 1, (len(idx), 1, 1)).astype("float32")
        point = tf.Variable(b + a * (xi[idx] - b))
        extra = [] if k is None else [tf.constant(k[i0:i0 + rows_per][idx]), tf.constant(s[i0:i0 + rows_per][idx])]
        for j in range(n_out):
            with tf.GradientTape() as tape:
                y = model([point] + extra if extra else point, training=False)
                total = tf.reduce_sum(y[:, j])
            g = tape.gradient(total, point).numpy()
            out[j, i0:i0 + len(xi)] = ((xi[idx] - b) * g).reshape(len(xi), nsamples, *x.shape[1:]).mean(axis=1)
    return out


def explain(cfg, method, model_name, direct=None):
    """SHAP values summed over the 48 input hours, plus the checks, for one run x model.

    What: returns (shap array (outputs, windows, features), latest-hour inputs (windows, features),
          output list, feature names, check text).
    How:  shap.GradientExplainer(model, train background) for the single-input models; the direct formula for
          the TFT (or when direct=True); checks that the wrapped model gives the saved median forecast and how
          closely SHAP adds up to f(x) - E f(background).
    """
    s_cfg = cfg["shap"]
    rc = run_config(method)
    model, outs = wrapped_model(cfg, rc, model_name)
    many = len(model.inputs) > 1
    direct = many if direct is None else direct
    bg_x, bg_k, bg_s, _ = windows(rc, "train", s_cfg["background"], cfg["seed"])
    x, k, s, rows = windows(rc, "calib", s_cfg["explain"], cfg["seed"] + 1)
    feed = (lambda a, kk, ss: [a, kk, ss]) if many else (lambda a, kk, ss: a)
    fx = model.predict(feed(x, k, s), verbose=0, batch_size=s_cfg["batch_size"])
    saved = pd.read_csv(rc["forecasts_dir"] / model_name / "quantile" / "calibration.csv").iloc[rows]
    gap_saved = max(np.nanmax(np.abs(fx[:, j] - saved[f"median_{short(t)}_h{h}"].to_numpy())) for j, (t, h) in
                    enumerate(outs))
    if direct:
        sv = direct_expected_gradients(model, x, k if many else None, s if many else None, bg_x,
                                       s_cfg["nsamples"], s_cfg["direct_batch"], cfg["seed"])
        expected = model.predict(feed(bg_x, bg_k, bg_s), verbose=0).mean(axis=0) if not many else None
    else:
        sv = shap.GradientExplainer(model, bg_x, batch_size=s_cfg["batch_size"]).shap_values(
            x, nsamples=s_cfg["nsamples"], rseed=cfg["seed"])
        sv = np.stack(sv) if isinstance(sv, list) else np.moveaxis(sv, -1, 0)  # (outputs, windows, 48, features)
        expected = model.predict(bg_x, verbose=0).mean(axis=0)
    per_input = sv.sum(axis=2)                                               # sum over the 48 input hours
    names = json.loads((rc["samples_dir"] / "feature_names.json").read_text(encoding="utf-8"))
    check = f"wrapped model vs saved median forecast: largest difference {gap_saved:.4f} µg/m³"
    if expected is not None:
        gap_add = np.abs(per_input.sum(axis=2) - (fx.T - expected[:, None]))
        check += (f"; SHAP additivity: median gap {np.median(gap_add):.2f} µg/m³ vs mean |forecast - average| "
                  f"{np.mean(np.abs(fx - expected)):.2f} µg/m³")
    else:
        check += "; additivity not defined here (calendar and station held at each window's own values)"
    return per_input, x[:, -1, :], outs, names, check


def importance_table(per_input, outs, names):
    """Mean |SHAP| and mean SHAP (µg/m³) per input, target and horizon, ranked within each output."""
    rows = []
    for k, (t, h) in enumerate(outs):
        for f, name in enumerate(names):
            rows.append({"target": t, "horizon": h, "feature": name,
                         "mean_abs_shap": np.abs(per_input[k, :, f]).mean(), "mean_shap": per_input[k, :, f].mean()})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby(["target", "horizon"])["mean_abs_shap"].rank(ascending=False).astype(int)
    return out.sort_values(["target", "horizon", "rank"])


def top_inputs(table, target, n):
    """The n inputs with the largest mean |SHAP| for one target, averaged over every explained horizon."""
    t = table[table["target"] == target]
    return t.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(n).index.tolist()


def importance_chart(table, cfg, path, title):
    """Mean |SHAP| as a heatmap: one panel per target, rows = its top inputs, columns = every horizon.

    What: the value (µg/m³) is written in each cell; darker blue = larger effect (one hue, own scale per target).
    How:  rows ranked by the mean over the horizons (top_inputs).
    """
    s_cfg, hs = cfg["shap"], cfg["shap"]["horizons"]
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(15, 7.2), facecolor=SURFACE)
    for ax, target in zip(axes, cfg["targets"]):
        top = top_inputs(table, target, s_cfg["top"])
        grid = table[table["target"] == target].pivot(index="feature", columns="horizon",
                                                     values="mean_abs_shap").loc[top, hs]
        im = ax.imshow(grid.to_numpy(), cmap=HEAT, vmin=0, aspect="auto")
        for i in range(len(top)):
            for j in range(len(hs)):
                v = grid.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > 0.6 * grid.to_numpy().max() else INK)
        ax.set_xticks(range(len(hs)), [f"+{h} h" for h in hs])
        ax.set_yticks(range(len(top)), top)
        ax.tick_params(length=0, labelcolor=INK)
        ax.xaxis.tick_top()
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(target, loc="left", fontsize=11, color=INK, pad=24)
        bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("mean |SHAP| (µg/m³)", color=MUTED)
        bar.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.925, f"Mean absolute SHAP value over {s_cfg['explain']} random calibration windows (May–Aug "
                          "2025), in µg/m³ of forecast change; each input's effect is summed over its 48 input hours. "
                          "Darker = the input moves the forecast more.", fontsize=9, color=SECONDARY)
    fig.subplots_adjust(top=0.82, wspace=0.6, left=0.13, right=0.97)
    save(fig, path)


def beeswarm_chart(per_input, latest, outs, names, table, target, cfg, path, title):
    """SHAP beeswarm for one target: one panel per horizon, the same top inputs as rows in every panel.

    What: x = SHAP value of one window (µg/m³), rows = top inputs averaged over the horizons; colour = the
          input's value at the latest input hour, as its rank among the explained windows (blue low, red high).
    How:  vertical spread of the dots follows how many windows share a SHAP value (a simple swarm).
    """
    s_cfg = cfg["shap"]
    rng = np.random.default_rng(cfg["seed"])
    rank = pd.DataFrame(latest).rank(pct=True).to_numpy()
    top = top_inputs(table, target, s_cfg["top"])
    fig, axes = plt.subplots(1, len(s_cfg["horizons"]), figsize=(4.1 * len(s_cfg["horizons"]) + 2.5, 6.6),
                             facecolor=SURFACE, sharey=True, sharex=True)
    for ax, h in zip(axes, s_cfg["horizons"]):
        k = outs.index((target, h))
        for row, name in enumerate(top[::-1]):
            f = names.index(name)
            v = per_input[k, :, f]
            counts, edges = np.histogram(v, bins=40)
            dens = counts[np.clip(np.digitize(v, edges) - 1, 0, len(counts) - 1)] / counts.max()
            y = row + rng.uniform(-1, 1, len(v)) * 0.38 * dens
            sc = ax.scatter(v, y, c=rank[:, f], cmap=VALUE_CMAP, vmin=0, vmax=1, s=9, alpha=0.85, linewidths=0)
        ax.axvline(0, color=RULE, linewidth=1, zorder=0)
        ax.set_yticks(range(len(top)), top[::-1])
        style(ax, f"+{h} h ahead", "")
        ax.set_xlabel("SHAP value (µg/m³)", color=MUTED)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    bar = fig.colorbar(sc, cax=fig.add_axes((0.945, 0.2, 0.008, 0.55)), ticks=[0, 0.5, 1])   # own space, right
    bar.ax.set_yticklabels(["low", "middle", "high"], color=MUTED)
    bar.set_label("input value at the latest hour", color=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.905, "One dot = one calibration window. Right of the line = the input pushed the forecast up; "
                          "left = down. Colour = how high the input was in that window.", fontsize=9, color=SECONDARY)
    fig.text(0.01, 0.873, "0 = the average forecast over the 2023–24 training windows (all seasons, smoggy winters "
                          "included). May–Aug is the clean monsoon, so most dots sit left of 0.", fontsize=9,
             color=SECONDARY)
    fig.subplots_adjust(left=0.1, right=0.92, top=0.8, bottom=0.11, wspace=0.08)
    save(fig, path)


def correlation_chart(latest, names, table, cfg, path, title):
    """Pearson correlation between the most important inputs (latest input hour), lower triangle with values.

    What: inputs = top by mean |SHAP| share, averaged over every explained target and horizon.
    How:  each output's mean |SHAP| is divided by its total first, so PM10's larger µg/m³ values do not dominate.
    """
    share = table.assign(share=table["mean_abs_shap"] / table.groupby(["target", "horizon"])["mean_abs_shap"]
                         .transform("sum")).groupby("feature")["share"].mean().sort_values(ascending=False)
    top = share.head(cfg["shap"]["top"]).index.tolist()
    r = np.corrcoef(latest[:, [names.index(n) for n in top]], rowvar=False)
    mask = np.triu(np.ones_like(r, dtype=bool))           # upper triangle and the diagonal of 1s left out
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    im = ax.imshow(np.ma.array(r, mask=mask)[1:, :-1], cmap=CORR_CMAP, vmin=-1, vmax=1)
    for i in range(1, len(top)):
        for j in range(i):
            ax.text(j, i - 1, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(r[i, j]) > 0.75 else INK)
    labels = top
    ax.set_xticks(range(len(top) - 1), labels[:-1], rotation=45, ha="right")
    ax.set_yticks(range(len(top) - 1), labels[1:])
    ax.tick_params(length=0, labelcolor=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Pearson r (−1 opposite · 0 none · +1 together)", color=MUTED)
    bar.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=28)
    ax.text(0, 1.02, "The 12 inputs that matter most (SHAP), measured on the explained calibration windows at their "
                     "latest input hour.", transform=ax.transAxes, fontsize=9, color=SECONDARY)
    save(fig, path)


GROUPS = {"PM levels (now + lags)": "#2a78d6", "PM averages": "#eb6834", "PM spreads": "#1baf7a",
          "season (day of year)": "#eda100", "weather": "#e87ba4", "time of day": "#008300", "gases": "#6250d6",
          "stuck flags": "#e34948"}                       # validated categorical order (dataviz), fixed per group


def group_of(name):
    """Input name -> one of GROUPS."""
    base, _, part = name.partition("_")
    if name.startswith("stuck_"):
        return "stuck flags"
    if base in ("PM2.5", "PM10"):
        return "PM averages" if part.startswith("mean") else "PM spreads" if part.startswith("std") else "PM levels (now + lags)"
    if base == "doy":
        return "season (day of year)"
    if base == "hour":
        return "time of day"
    return "weather" if base in ("RH", "SR", "BP", "AT", "RF", "WD") else "gases"


def share_chart(table, cfg, path, title):
    """Share of the total SHAP effect: left = per input group and horizon (100% bars), right = all inputs ranked.

    What: share = an input's mean |SHAP| divided by the sum over all inputs, per target and horizon; left
          averages the two targets per horizon, right averages all explained targets and horizons.
    How:  groups by meaning (group_of); one colour per group, the same in both panels.
    """
    t = table.assign(group=table["feature"].map(group_of),
                     share=table["mean_abs_shap"] / table.groupby(["target", "horizon"])["mean_abs_shap"].transform("sum"))
    by_h = t.groupby(["horizon", "group"])["share"].sum().unstack().div(len(cfg["targets"]))[list(GROUPS)] * 100
    ranked = t.groupby(["feature", "group"])["share"].mean().reset_index().sort_values("share")
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 12), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    hs = cfg["shap"]["horizons"]
    for y, h in enumerate(hs[::-1]):
        start = 0.0
        for g, color in GROUPS.items():
            v = by_h.loc[h, g]
            left.barh(y, v, left=start, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=g)
            if v >= 5:
                left.text(start + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                          color="#0b0b0b" if color == "#eda100" else "#ffffff", fontweight="bold")
            start += v
    left.set_yticks(range(len(hs)), [f"+{h} h" for h in hs[::-1]])
    left.set_xlim(0, 100)
    left.tick_params(axis="y", length=0, labelcolor=INK)
    style(left, "Share of the effect per input group, by horizon (both targets)", "")
    left.set_xlabel("% of the total SHAP effect", color=MUTED)
    right.barh(list(ranked["feature"]), ranked["share"] * 100, height=0.7,
               color=[GROUPS[g] for g in ranked["group"]])
    for y, v in enumerate(ranked["share"] * 100):
        right.text(v + 0.05, y, f"{v:.1f}", va="center", fontsize=6.5, color=SECONDARY)
    right.tick_params(axis="y", length=0, labelcolor=INK, labelsize=7.5)
    style(right, "All 50 inputs, ranked (mean over both targets and 5 horizons)", "")
    right.set_xlabel("% of the total SHAP effect", color=MUTED)
    handles, labels = left.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    right.legend(keep.values(), keep.keys(), loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.95, "Share = an input's mean |SHAP| ÷ the sum over all 50 inputs (100% per target and horizon). "
                         f"{cfg['shap']['explain']} calibration windows, May–Aug 2025.", fontsize=9, color=SECONDARY)
    fig.subplots_adjust(top=0.91, wspace=0.42, left=0.05)
    save(fig, path)


def share_all_chart(all_table, cfg, path):
    """Share of the SHAP effect per input group for every run x model: heatmaps at the first and last horizon.

    What: rows = imputation run x model, columns = input groups; cell = % of the total effect (both targets);
          one-hue blue, value written in every cell.
    How:  share per target and horizon, summed per group, averaged over the targets.
    """
    t = all_table.assign(group=all_table["feature"].map(group_of))
    t["share"] = t["mean_abs_shap"] / t.groupby(["method", "model", "target", "horizon"])["mean_abs_shap"].transform("sum")
    methods, models = cfg["imputation"]["methods"], cfg["forecasters"]["names"]
    rows = [(m, n) for m in methods for n in models]
    hs = [cfg["shap"]["horizons"][0], cfg["shap"]["horizons"][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(17, 13), facecolor=SURFACE)
    for ax, h in zip(axes, hs):
        g = t[t["horizon"] == h].groupby(["method", "model", "group"])["share"].sum().div(len(cfg["targets"])) * 100
        grid = g.unstack().reindex(index=pd.MultiIndex.from_tuples(rows), columns=list(GROUPS)).to_numpy()
        ax.imshow(grid, cmap=HEAT, vmin=0, vmax=np.nanmax(grid), aspect="auto")
        for i in range(len(rows)):
            for j in range(len(GROUPS)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.6 * np.nanmax(grid) else INK)
        ax.set_xticks(range(len(GROUPS)), list(GROUPS), rotation=35, ha="left")
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), [f"{m} · {LABELS[n]}" for m, n in rows], fontsize=8)
        for k in range(1, len(methods)):
            ax.axhline(k * len(models) - 0.5, color=SURFACE, linewidth=3)
        ax.tick_params(length=0, labelcolor=INK)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"+{h} h ahead", loc="left", fontsize=12, color=INK, pad=110)
    fig.suptitle("Where the forecast's information comes from: every imputation run and model", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.955, "Cell = % of the total SHAP effect from that input group (both targets, "
                          f"{cfg['shap']['explain']} calibration windows each). Rows add up to 100%. TFT: the 48-hour "
                          "window only (its calendar and station inputs are not explained).", fontsize=9, color=SECONDARY)
    fig.subplots_adjust(top=0.78, wspace=0.45, left=0.12, right=0.98)
    save(fig, path)


def pairs(cfg):
    """(run, model) pairs to explain: all, or those in the environment variable AQI_SHAP_ONLY ("ffill:gru,knn:tft")."""
    only = os.environ.get("AQI_SHAP_ONLY")
    if only:
        return [tuple(p.split(":")) for p in only.split(",")]
    return [(m, n) for m in cfg["imputation"]["methods"] for n in cfg["forecasters"]["names"]]


def main():
    """Explain the published forecast of every run x model and save the SHAP tables and charts.

    What: per pair, figures_dir/runs/<run>/<model>/shap_importance.csv and the importance, beeswarm (one per
          target), correlation and share charts; then figures_dir/shap_importance_all.csv and the summary chart;
          first a check that the direct expected-gradients code agrees with the shap library.
    How:  explain -> importance_table -> the chart functions, pair by pair (progress printed).
    """
    cfg = load_config()
    s_cfg = cfg["shap"]
    lines, tables, done = [], [], {}
    if not os.environ.get("AQI_SHAP_ONLY"):
        pair = (s_cfg["check_run"], s_cfg["check_model"])
        done[pair] = explain(cfg, *pair, direct=False)                       # reused in the loop below
        lib, own = done[pair], explain(cfg, *pair, direct=True)
        a, b = (importance_table(r[0], r[2], r[3]) for r in (lib, own))
        m = a.merge(b, on=["target", "horizon", "feature"])
        rho = m.groupby(["target", "horizon"]).apply(lambda g: spearmanr(g["mean_abs_shap_x"], g["mean_abs_shap_y"])[0])
        lines.append(f"check: direct expected gradients vs shap library ({s_cfg['check_model']}, {s_cfg['check_run']}): "
                     f"rank correlation of importance {rho.min():.3f}–{rho.max():.3f}")
        print(lines[-1], flush=True)
    for method, model_name in pairs(cfg):
        out_dir = run_config(method)["figures_dir"] / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        per_input, latest, outs, names, check = done.get((method, model_name)) or explain(cfg, method, model_name)
        table = importance_table(per_input, outs, names)
        table.round(4).to_csv(out_dir / s_cfg["file"], index=False)
        tables.append(table.assign(method=method, model=model_name))
        who = f"{LABELS[model_name]}, {method} run"
        importance_chart(table, cfg, out_dir / s_cfg["importance_plot"], f"Which inputs drive the forecast ({who})")
        for target in cfg["targets"]:
            beeswarm_chart(per_input, latest, outs, names, table, target, cfg,
                           out_dir / s_cfg["beeswarm_plot"].format(target=short(target)),
                           f"How each input pushes the {target} forecast up or down ({who})")
        correlation_chart(latest, names, table, cfg, out_dir / s_cfg["correlation_plot"],
                          f"How the most important inputs move together ({who})")
        share_chart(table, cfg, out_dir / s_cfg["share_plot"], f"Where the forecast's information comes from ({who})")
        lines.append(f"{method} · {model_name}: {check}")
        print(lines[-1], flush=True)
    all_table = pd.concat(tables, ignore_index=True)
    if not os.environ.get("AQI_SHAP_ONLY"):
        all_table.round(4).to_csv(cfg["figures_dir"] / s_cfg["all_file"], index=False)
        share_all_chart(all_table, cfg, cfg["figures_dir"] / s_cfg["share_all_plot"])
    save_text(cfg, "shap_report.txt", "\n".join(lines))


if __name__ == "__main__":
    main()
