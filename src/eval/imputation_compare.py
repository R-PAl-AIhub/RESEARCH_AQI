"""Compare the imputation methods (one full run each): samples, calibration errors, Worker B and test coverage.

Ranking uses CALIBRATION errors only (freeze rule); test numbers are reported, never used to pick.
Methods that use future values (config imputation.leaks_future) are shown but never picked.
"""
import os

import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure
from src.models.compare import COLORS, LABELS
from src.temp_out import save_text

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
SPLITS = ("train", "val", "calib", "test")


def run_config(method):
    """Settings of one run (the folders of AQI_IMPUTATION=<method>)."""
    os.environ["AQI_IMPUTATION"] = method
    try:
        return load_config()
    finally:
        os.environ.pop("AQI_IMPUTATION")


def model_row(cfg, model):
    """All numbers of one method x model, as one dict.

    What: calibration RMSE per target and horizon (step 4), condition number (5.2),
          s_raw vs s_embed correlation (5.6), calibration Green and check-week Red
          shares (5.6b), test Red share (7.1), PM2.5 and PM10 test coverage of split
          conformal and CQR + adaptive on normal and smoggiest-5% hours (7.2): min, max
          and mean over the horizons (PM10 columns end in _pm10).
    How:  reads the files those steps wrote in this run's folders.
    """
    fig, art, tmp = cfg["figures_dir"], cfg["forecasts_dir"], cfg["temp_dir"]
    row = {}
    errors = pd.read_csv(fig / cfg["forecast_compare"]["errors_file"])
    for (target, h), v in errors[errors["model"] == model].set_index(["target", "horizon"])["rmse"].items():
        row[f"calib_rmse_{target}_h{h}"] = v
    row["cond_max"] = pd.read_csv(tmp / f"5.2_{model}_season_shapes.csv")["condition"].max()
    row["r_raw_embed"] = pd.read_csv(tmp / "5.6_score_compare.csv").set_index("model").loc[model, "pearson"]
    labels = pd.read_csv(art / model / "worker_b" / cfg["worker_b"]["labels_file"],
                         usecols=lambda c: c.startswith("label_"))
    row["calib_green"] = (labels == "Green").to_numpy().mean()
    check = pd.read_csv(tmp / f"5.6b_{model}_check_shares.csv")
    check = check[check["season"] == "all"]["Red"]
    row["check_red_min"], row["check_red_max"] = check.min(), check.max()
    flags = pd.read_csv(art / model / cfg["evaluation"]["flags_file"], usecols=lambda c: c.startswith("flag_"))
    row["test_red"] = (flags == "Red").to_numpy().mean()
    t1 = pd.read_csv(fig / model / cfg["evaluation"]["table1_file"])
    for target, end in (("PM2.5", ""), ("PM10", "_pm10")):            # PM2.5 keeps the original column names
        tt = t1[t1["target"] == target]
        for method, short in (("(a) split conformal", "split"), ("(c) CQR + adaptive", "adaptive")):
            for group in ("normal", "extreme"):
                cov = tt[(tt["method"] == method) & (tt["group"] == group)]["coverage"]
                for stat in ("min", "max", "mean"):
                    row[f"cov_{short}_{group}{end}_{stat}"] = cov.agg(stat)
    return row


def compare_table(cfg):
    """One row per method x model (methods whose run is incomplete are skipped with a note)."""
    rows, notes = [], []
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        try:
            samples = {s: len(pd.read_csv(rc["samples_dir"] / f"meta_{s}.csv", usecols=["station"])) for s in SPLITS}
        except FileNotFoundError as err:
            notes.append(f"{method}: no samples ({err.filename})")
            continue
        for model in cfg["forecasters"]["names"]:
            try:
                rows.append({"method": method, "model": model,
                             "uses_future": method in cfg["imputation"]["leaks_future"],
                             **{f"samples_{s}": n for s, n in samples.items()}, **model_row(rc, model)})
            except FileNotFoundError as err:
                notes.append(f"{method} / {model}: incomplete run ({err.filename})")
    return pd.DataFrame(rows), notes


def common_rmse(cfg):
    """Calibration RMSE of every method x model on the SAME calibration samples (those all methods kept).

    What: one row per method x model: mean RMSE over all horizons per target, and over both targets.
    How:  the point forecasts (calib_pred.csv) and the real values (meta_calib.csv) are
          matched on station + timestamp; only keys present in every method's run count,
          so deletion (which keeps only complete windows) is not scored on easier hours.
    """
    runs = {}
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        runs[method] = (rc, pd.read_csv(rc["samples_dir"] / "meta_calib.csv").set_index(["station", "timestamp"]))
    keys = sorted(set.intersection(*(set(meta.index) for _, meta in runs.values())))
    rows = []
    for method, (rc, meta) in runs.items():
        truth = meta.loc[keys]
        for model in cfg["forecasters"]["names"]:
            pred = pd.read_csv(rc["forecasts_dir"] / model / "point" / "calib_pred.csv").set_index(
                ["station", "timestamp"]).loc[keys]
            row = {"method": method, "model": model, "common_samples": len(keys)}
            for target in cfg["targets"]:
                t = target.lower().replace(".", "")
                row[f"common_rmse_{target}"] = np.mean(
                    [np.sqrt(((pred[f"pred_{t}_h{h}"] - truth[f"{t}_ugm3_h{h}"]) ** 2).mean()) for h in cfg["horizons"]])
            row["common_rmse_mean"] = np.mean([row[f"common_rmse_{t}"] for t in cfg["targets"]])
            rows.append(row)
    return pd.DataFrame(rows)


def compare_chart(table, cfg, out):
    """Grouped bars per method (one bar per model): calibration RMSE and test coverage on the smoggiest hours.

    What: 2 x 2 panels, one bar per model for every method: top = calibration RMSE PM2.5
          and PM10; bottom = test coverage of the smoggiest 5% of hours (top 5% of the
          observed value, step 7.2), PM2.5 and PM10, with the 90% aim dashed.
    How:  RMSE = mean over the horizons on the calibration hours every method kept (common_rmse);
          coverage = CQR + adaptive (the published range), mean over the horizons.
          Test numbers are shown only; the pick uses calibration RMSE (freeze rule).
    """
    methods, models = list(dict.fromkeys(table["method"])), cfg["forecasters"]["names"]
    width = 0.8 / len(models)
    x = np.arange(len(methods))
    n = int(table["common_samples"].iloc[0])
    panels = [("common_rmse_PM2.5", f"calibration RMSE PM2.5, the same {n:,} hours for every method", "µg/m³"),
              ("common_rmse_PM10", f"calibration RMSE PM10, the same {n:,} hours for every method", "µg/m³"),
              ("cov_adaptive_extreme_mean", "test coverage PM2.5, smoggiest 5% of hours (CQR + adaptive)",
               "% inside the range"),
              ("cov_adaptive_extreme_pm10_mean", "test coverage PM10, smoggiest 5% of hours (CQR + adaptive)",
               "% inside the range")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=SURFACE)
    for ax, (key, title, unit) in zip(axes.ravel(), panels):
        cov = key.startswith("cov_")
        for k, model in enumerate(models):
            t = table[table["model"] == model].set_index("method").reindex(methods)
            ax.bar(x + (k - (len(models) - 1) / 2) * width, t[key] * (100 if cov else 1), width * 0.92,
                   color=COLORS[model], label=LABELS[model])
        if cov:
            ax.axhline((1 - cfg["alpha"]) * 100, color=INK, linestyle="--", linewidth=1, label="aim 90%")
        ax.set_xticks(x, methods)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylabel(unit, color=MUTED)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1, 1].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Imputation methods compared (each a full run 3.2 -> 7.2)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Build figures_dir/imputation_compare.csv and .png and print the calibration pick.

    What: the best method = lowest mean calibration RMSE (all models, both targets,
          all horizons) among the methods that do not use future values.
    How:  reads every run's outputs; test numbers are printed but never used to pick.
    """
    cfg = load_config()
    table, notes = compare_table(cfg)
    table = table.merge(common_rmse(cfg), on=["method", "model"], how="left")
    imp = cfg["imputation"]
    table.round(4).to_csv(cfg["figures_dir"] / imp["compare_file"], index=False)
    compare_chart(table, cfg, cfg["figures_dir"] / imp["compare_plot_file"])
    rmse_cols = [c for c in table.columns if c.startswith("calib_rmse_")]
    score = table.assign(own_rmse=table[rmse_cols].mean(axis=1)).groupby("method").agg(
        own_samples_rmse=("own_rmse", "mean"), common_samples_rmse=("common_rmse_mean", "mean"),
        calib_samples=("samples_calib", "first"), common_samples=("common_samples", "first"),
        uses_future=("uses_future", "first"), test_red=("test_red", "mean"))
    allowed = score[~score["uses_future"]]
    best = allowed["common_samples_rmse"].idxmin() if len(allowed) else "none"
    lines = ["calibration RMSE per method (mean over models, targets and horizons):",
             "own_samples_rmse = on each run's own samples; common_samples_rmse = on the samples every run kept",
             score.round(3).to_string(),
             "",
             f"best method (common calibration samples; methods using future values excluded): {best}",
             *notes]
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "imputation_compare_report.txt", text)


if __name__ == "__main__":
    main()
