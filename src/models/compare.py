"""Part 4 comparison: error tables and charts for the 5 forecasters (calibration May-Aug 2025 only)."""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure
from src.models.metrics import error_rows
from src.models.train import coverage, lines_array, load, load_split, short

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
LABELS = {"gru": "GRU", "lstm": "LSTM", "cnn": "1D CNN", "cnn_lstm": "CNN + LSTM", "tft": "TFT"}
COLORS = {"gru": "#2a78d6", "lstm": "#eb6834", "cnn": "#1baf7a", "cnn_lstm": "#eda100", "tft": "#e87ba4"}  # colourblind-checked order (dataviz validator, 2026-11-02)
BAND_LIGHT = "#b7d3f6"
PERIODS = {"train": "train 2023-24", "validation": "validation Jan-Apr 2025",
           "calibration": "calibration May-Aug 2025"}          # option C; test stays out of Part 4
PERIOD_COLORS = {"train": "#2a78d6", "validation": "#8a8f98", "calibration": "#eb6834"}


def style(ax, title, ylabel):
    """Quiet axes so the data stands out.

    What: title, y label, light horizontal grid, no top/right frame.
    How:  matplotlib axis settings; muted grey for frame and ticks.
    """
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def point_predictions(cfg, names, meta):
    """Calibration predictions of every point model, lined up with the calibration samples.

    What: returns {model: array (samples, targets, horizons)} in µg/m³.
    How:  reads forecasts_dir/<model>/point/calib_pred.csv (same row order as
          meta_calib.csv, checked on station and timestamp).
    """
    preds = {}
    for name in names:
        table = pd.read_csv(cfg["forecasts_dir"] / name / "point" / "calib_pred.csv", parse_dates=["timestamp"])
        assert table["station"].equals(meta["station"]) and table["timestamp"].equals(meta["timestamp"])
        preds[name] = lines_array(table, "pred", cfg["targets"], cfg["horizons"])
    return preds


def error_panels(errors, targets, horizons, path):
    """Grid of line charts: rows = targets, columns = RMSE, MSE, MAE; one line per model.

    What: x = forecast horizon, y = error on calibration.
    How:  reads the long errors table (model, target, horizon, rmse, mse, mae).
    """
    metrics = [("rmse", "µg/m³"), ("mse", "(µg/m³)²"), ("mae", "µg/m³")]
    fig, axes = plt.subplots(len(targets), 3, figsize=(16, 4.2 * len(targets)), facecolor=SURFACE)
    for r, target in enumerate(targets):
        for c, (metric, unit) in enumerate(metrics):
            ax = axes[r, c]
            for name, g in errors[errors["target"] == target].groupby("model", sort=False):
                ax.plot(g["horizon"], g[metric], marker="o", linewidth=2, color=COLORS[name], label=LABELS[name])
            ax.set_xticks(horizons, [f"+{h}h" for h in horizons])
            ax.set_xlabel("forecast horizon", color=MUTED)
            style(ax, f"{target} - {metric.upper()} (lower is better)", unit)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Part 4 - forecast errors on calibration (May-Aug 2025), point models",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def loss_panels(cfg, names, path):
    """Training vs validation loss per epoch: rows = point (MSE) and quantile (pinball), columns = models.

    What: two lines per panel and a dot on the best validation epoch (the weights kept).
    How:  reads forecasts_dir/<model>/<kind>/history.csv.
    """
    kinds = [("point", "MSE loss (scaled units)"), ("quantile", "pinball loss (scaled units)")]
    fig, axes = plt.subplots(2, len(names), figsize=(max(4 * len(names), 9), 7), facecolor=SURFACE, squeeze=False)
    for r, (kind, ylabel) in enumerate(kinds):
        for c, name in enumerate(names):
            ax = axes[r, c]
            h = pd.read_csv(cfg["forecasts_dir"] / name / kind / "history.csv", index_col="epoch")
            epochs = h.index + 1
            ax.plot(epochs, h["loss"], color=COLORS[name], linewidth=2, label="training")
            ax.plot(epochs, h["val_loss"], color=MUTED, linewidth=2, label="validation (Jan-Apr 2025)")
            best = h["val_loss"].idxmin()
            ax.plot(best + 1, h["val_loss"][best], "o", markersize=7, color=INK)
            ax.annotate(f"best: epoch {best + 1}", (best + 1, h["val_loss"][best]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=INK)
            ax.set_xlabel("epoch", color=MUTED)
            style(ax, f"{LABELS[name]} - {kind}", ylabel if c == 0 else "")
            if c == 0:
                ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Part 4 - training curves (same settings for every model)", color=INK, fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def coverage_panels(cover, targets, horizons, names, alpha, path):
    """Grouped bars: % of real values inside the low-high band; rows = targets, columns = train / calibration.

    What: one bar per model at each horizon, dashed line at the aim (1 - alpha).
    How:  cover is indexed by (model, split, target, horizon).
    """
    splits = ["train", "calibration"]
    fig, axes = plt.subplots(len(targets), 2, figsize=(15, 4.3 * len(targets)), facecolor=SURFACE)
    width = 0.8 / len(names)
    x = np.arange(len(horizons))
    for r, target in enumerate(targets):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            for k, name in enumerate(names):
                values = [cover[(name, split, target, h)] * 100 for h in horizons]
                bars = ax.bar(x + (k - (len(names) - 1) / 2) * width, values, width * 0.92,
                              color=COLORS[name], label=LABELS[name])
                ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, color=INK)
            ax.axhline((1 - alpha) * 100, color=INK, linestyle="--", linewidth=1, label=f"aim {1 - alpha:.0%}")
            ax.set_xticks(x, [f"+{h}h" for h in horizons])
            ax.set_ylim(0, 105)
            ax.set_xlabel("forecast horizon", color=MUTED)
            style(ax, f"{target} - {PERIODS[split]}", "% inside the band")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="lower left")
    fig.suptitle("Part 4 - quantile models: real values inside the low-high band", color=INK, fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def band_panels(tables, target, horizons, names, c_cfg, alpha, path):
    """Quantile bands for one target: rows = models, columns = horizons, one station, a short calibration window.

    What: real value (dark line), median (model colour) and the low-high band
          (light fill), all at the hour being forecast.
    How:  x = timestamp + h; hours without a sample are put back as empty rows
          so a gap shows as a break.
    """
    start = pd.Timestamp(c_cfg["band_start"])
    end = start + pd.Timedelta(days=c_cfg["band_days"])
    fig, axes = plt.subplots(len(names), len(horizons), figsize=(6 * len(horizons), 2.6 * len(names)),
                             facecolor=SURFACE, sharex=True, sharey=True, squeeze=False)
    for r, name in enumerate(names):
        t = tables[name]
        t = t[(t["station"] == c_cfg["band_station"]) & (t["timestamp"] >= start) & (t["timestamp"] < end)]
        for c, h in enumerate(horizons):
            ax = axes[r, c]
            k = f"{short(target)}_h{h}"
            w = t.assign(when=t["timestamp"] + pd.Timedelta(hours=h)).set_index("when")
            w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
            ax.fill_between(w.index, w[f"low_{k}"], w[f"high_{k}"], color=BAND_LIGHT, linewidth=0,
                            label=f"{1 - alpha:.0%} band")
            ax.plot(w.index, w[f"median_{k}"], color=COLORS[name], linewidth=1, label="median forecast")
            ax.plot(w.index, w[f"target_{k}"], color=INK, linewidth=0.8, label=f"real {target}")
            style(ax, f"{LABELS[name]}  +{h}h", "µg/m³" if c == 0 else "")
            ax.tick_params(axis="x", labelrotation=30)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    station = c_cfg["band_station"].replace(", Kolkata - WBPCB", "")
    fig.suptitle(f"Part 4 - {target} quantile bands, {station}, {c_cfg['band_days']} days from "
                 f"{c_cfg['band_start']} (calibration)", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def station_year_errors(cfg, model):
    """Median-line RMSE of one quantile forecaster per station, split, target and horizon.

    What: returns a long table: station, period (train / validation / calibration),
          target, horizon, rmse, mae, samples. Test is not read (Part 7 only).
    How:  reads the exported tables of the three splits and groups each by station.
    """
    rows = []
    for split in PERIODS:
        t = load(model, split)
        for station, g in t.groupby("station"):
            for target in cfg["targets"]:
                for h in cfg["horizons"]:
                    k = f"{short(target)}_h{h}"
                    err = g[f"median_{k}"] - g[f"target_{k}"]
                    rows.append({"station": station.replace(", Kolkata - WBPCB", ""), "period": split,
                                 "target": target, "horizon": h, "rmse": float(np.sqrt((err ** 2).mean())),
                                 "mae": float(err.abs().mean()), "samples": len(g)})
    return pd.DataFrame(rows).round(2)


def station_year_chart(table, model, target, horizons, path):
    """One model, one target: RMSE per station with one bar per split, one panel per horizon.

    What: grouped bars (train 2023-24, validation Jan-Apr 2025, calibration May-Aug 2025), value on top.
    How:  values from station_year_errors.
    """
    fig, axes = plt.subplots(len(horizons), 1, figsize=(15, 3.3 * len(horizons)), facecolor=SURFACE, sharex=True)
    t = table[table["target"] == target]
    stations = sorted(t["station"].unique())
    x, periods = np.arange(len(stations)), [p for p in PERIODS if p in set(t["period"])]
    width = 0.8 / len(periods)
    for ax, h in zip(axes, horizons):
        e = t[t["horizon"] == h].pivot(index="station", columns="period", values="rmse").loc[stations]
        for k, period in enumerate(periods):
            bars = ax.bar(x + (k - (len(periods) - 1) / 2) * width, e[period], width * 0.92,
                          color=PERIOD_COLORS[period], label=PERIODS[period])
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, color=INK)
        style(ax, f"+{h} h", "RMSE µg/m³")
    axes[-1].set_xticks(x, stations, rotation=15)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3)
    fig.suptitle(f"Part 4 - {LABELS[model]}: {target} median-forecast RMSE per station and split",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Build the Part 4 comparison tables and charts.

    What: figures_dir gets forecast_errors.csv (RMSE/MSE/MAE per model, target,
          horizon), forecast_quantile.csv (median RMSE and band coverage of the
          quantile models) and four charts (errors, loss, coverage, one band chart
          per target). Calibration and train only.
    How:  every calibration sample is scored (step 3.3 keeps only samples with all answers).
    """
    cfg = load_config()
    c_cfg, targets, horizons, alpha = cfg["forecast_compare"], cfg["targets"], cfg["horizons"], cfg["alpha"]
    names, fig_dir = cfg["forecasters"]["names"], cfg["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    calib = load_split(cfg["samples_dir"], "calib")
    preds = point_predictions(cfg, names, calib["meta"])
    errors = error_rows(calib["y_raw"], preds, targets, horizons)
    errors.to_csv(fig_dir / c_cfg["errors_file"], index=False)

    tables = {n: load(n, "calibration") for n in names}
    cover, rows = {}, []
    for name in names:
        for split, table in (("train", load(name, "train")), ("calibration", tables[name])):
            for (target, h), v in coverage(table, targets, horizons).items():
                cover[(name, split, target, h)] = v
        truth, med = (lines_array(tables[name], k, targets, horizons) for k in ("target", "median"))
        med_err = error_rows(truth, {name: med}, targets, horizons)
        rows += [{**r, "train_coverage": round(cover[(name, "train", r["target"], r["horizon"])], 3),
                  "calib_coverage": round(cover[(name, "calibration", r["target"], r["horizon"])], 3)}
                 for r in med_err.rename(columns={"rmse": "median_rmse"})[
                     ["model", "target", "horizon", "median_rmse"]].to_dict("records")]
    pd.DataFrame(rows).to_csv(fig_dir / c_cfg["quantile_file"], index=False)

    error_panels(errors, targets, horizons, fig_dir / c_cfg["errors_plot"])
    loss_panels(cfg, names, fig_dir / c_cfg["loss_plot"])
    coverage_panels(cover, targets, horizons, names, alpha, fig_dir / c_cfg["coverage_plot"])
    for target in targets:
        band_panels(tables, target, horizons, names, c_cfg, alpha, fig_dir / c_cfg["band_plot"].format(target=short(target)))
    for name in names:                                       # one set of files per model
        md = fig_dir / name
        md.mkdir(parents=True, exist_ok=True)
        own = errors[errors["model"] == name]
        error_panels(own, targets, horizons, md / c_cfg["errors_plot"])
        loss_panels(cfg, [name], md / c_cfg["loss_plot"])
        coverage_panels(cover, targets, horizons, [name], alpha, md / c_cfg["coverage_plot"])
        sy = station_year_errors(cfg, name)
        sy.to_csv(fig_dir / name / c_cfg["station_year_file"], index=False)
        for target in targets:
            station_year_chart(sy, name, target, horizons,
                               fig_dir / name / c_cfg["station_year_plot"].format(target=short(target)))
            band_panels({name: tables[name]}, target, horizons, [name], c_cfg, alpha,
                        fig_dir / name / c_cfg["band_plot"].format(target=short(target)))

    print(f"calibration samples scored: {len(calib['y_raw'])}")
    print("\nRMSE (µg/m³) per model:")
    print(errors.pivot_table(index=["target", "horizon"], columns="model", values="rmse").to_string())
    print("\nquantile models:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nsaved to: {fig_dir}")


if __name__ == "__main__":
    main()
