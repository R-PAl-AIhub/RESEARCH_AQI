"""Teacher update (2026-11-02): accuracy table on every split, imputation run and model, and the R² chart.

Forecast = the published one: the median of the quantile model (the number step 7.1 publishes).
RMSE, MAE and R² per method, model, split, target and horizon; the summary is the mean over the horizons.
Test is reported only (freeze rule): nothing here chooses a setting.
"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import load_config
from src.eval.imputation_compare import run_config
from src.models.compare import COLORS, INK, LABELS, MUTED, SURFACE, style
from src.models.train import short
from src.plot_style import save_figure
from src.temp_out import save_text

SECONDARY, RULE, HAIRLINE = "#52514e", "#c3c2b7", "#e1e0d9"
SPLIT_NAMES = {"train": "Train 2023–24", "validation": "Validation Jan–Apr 2025",
               "calibration": "Calibration May–Aug 2025", "test": "Test Sep–Dec 2025"}
R2_SHADE = LinearSegmentedColormap.from_list("r2", ["#f3f7fd", "#86b6ef"])   # one hue, light -> darker = higher R²


def scores(y, p):
    """RMSE, MAE and R² of one forecast column against the real values.

    What: dict with n (rows used), rmse, mae (µg/m³) and r2 = 1 - SSE / SST.
    How:  rows where either value is empty are left out; SST around the mean of the real values.
    """
    keep = ~(np.isnan(y) | np.isnan(p))
    y, err = y[keep], p[keep] - y[keep]
    return {"n": int(keep.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def metric_rows(cfg):
    """One row per method x model x split x target x horizon.

    What: DataFrame with method, model, split, target, horizon, n, rmse, mae, r2.
    How:  reads each run's quantile table <model>/quantile/<split>.csv (target_ and median_ columns).
    """
    rows = []
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        for model in cfg["forecasters"]["names"]:
            for split in cfg["metrics"]["splits"]:
                table = pd.read_csv(rc["forecasts_dir"] / model / "quantile" / f"{split}.csv")
                for target in cfg["targets"]:
                    for h in cfg["horizons"]:
                        key = f"{short(target)}_h{h}"
                        rows.append({"method": method, "model": model, "split": split, "target": target, "horizon": h,
                                     **scores(table[f"target_{key}"].to_numpy(float),
                                              table[f"median_{key}"].to_numpy(float))})
    return pd.DataFrame(rows)


def summary_table(rows):
    """Mean over the horizons per method, model, split and target (n = rows of the +1 h column)."""
    keys = ["method", "model", "split", "target"]
    out = rows.groupby(keys, sort=False)[["rmse", "mae", "r2"]].mean()
    out["n"] = rows[rows["horizon"] == rows["horizon"].min()].set_index(keys)["n"]
    return out.reset_index()


def table_chart(summary, target, cfg, path):
    """The summary of one target as a clean table picture.

    What: rows = imputation run x model (grouped by run), columns = 4 splits x (RMSE, MAE, R²);
          R² cells shaded light -> darker blue by value (per split), the best R² of each split in bold.
    How:  plain matplotlib text and rectangles on an empty axes (no grid, hairline rules only).
    """
    s = summary[summary["target"] == target]
    methods, models, splits = cfg["imputation"]["methods"], cfg["forecasters"]["names"], cfg["metrics"]["splits"]
    col_w, gap, left = 0.9, 0.35, 3.3                     # data column width, gap between splits, label width
    x0 = {sp: left + i * (3 * col_w + gap) for i, sp in enumerate(splits)}
    width = left + len(splits) * (3 * col_w + gap)
    n_rows = len(methods) * len(models)
    fig, ax = plt.subplots(figsize=(width * 1.15, 0.34 * n_rows + 2.6), facecolor=SURFACE)
    ax.set_xlim(0, width)
    ax.set_ylim(n_rows + 2.2, -0.2)
    ax.axis("off")
    for sp in splits:                                     # header: split name and metric names
        ax.text(x0[sp] + 1.5 * col_w, 0.15, SPLIT_NAMES[sp], ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
        for j, name in enumerate(["RMSE", "MAE", "R²"]):
            ax.text(x0[sp] + (j + 0.5) * col_w, 0.95, name, ha="center", va="center", fontsize=9, color=SECONDARY)
        ax.plot([x0[sp] + 0.05, x0[sp] + 3 * col_w - 0.05], [0.55, 0.55], color=RULE, linewidth=0.8)
    ax.text(0.05, 0.95, "Imputation", va="center", fontsize=9, color=SECONDARY)
    ax.text(1.55, 0.95, "Model", va="center", fontsize=9, color=SECONDARY)
    ax.plot([0, width], [1.4, 1.4], color=RULE, linewidth=1)
    best = s[~s["method"].isin(cfg["metrics"]["not_best"])].groupby("split")["r2"].max()
    lo, hi = s.groupby("split")["r2"].min(), s.groupby("split")["r2"].max()
    y = 1.9
    for m_i, method in enumerate(methods):
        for k, model in enumerate(models):
            if k == 0:
                ax.text(0.05, y, method, va="center", fontsize=9.5, color=INK, fontweight="bold")
            ax.text(1.55, y, LABELS[model], va="center", fontsize=9.5, color=INK)
            for sp in splits:
                r = s[(s["method"] == method) & (s["model"] == model) & (s["split"] == sp)].iloc[0]
                shade = (r["r2"] - lo[sp]) / (hi[sp] - lo[sp]) if hi[sp] > lo[sp] else 1.0
                ax.add_patch(plt.Rectangle((x0[sp] + 2 * col_w + 0.06, y - 0.42), col_w - 0.12, 0.84,
                                           facecolor=R2_SHADE(shade), edgecolor="none"))
                for j, (val, fmt) in enumerate([(r["rmse"], "{:.1f}"), (r["mae"], "{:.1f}"), (r["r2"], "{:.3f}")]):
                    bold = j == 2 and method not in cfg["metrics"]["not_best"] and np.isclose(r["r2"], best[sp])
                    ax.text(x0[sp] + (j + 0.5) * col_w, y, fmt.format(val), ha="center", va="center", fontsize=9.5,
                            color=INK, fontweight="bold" if bold else "normal")
            y += 1
        if m_i < len(methods) - 1:
            ax.plot([0, width], [y - 0.5, y - 0.5], color=HAIRLINE, linewidth=0.8)
    fig.suptitle(f"Forecast accuracy of every model and imputation run: {target}", x=0.02, y=0.995, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.02, 0.972, "Published forecast = median of the quantile model · mean over the +1, +3, +6, +12 and "
                          "+24 h horizons · RMSE and MAE in µg/m³ (lower is better) · R² (higher is better, 1 = perfect)",
             fontsize=9.5, color=SECONDARY)
    fig.text(0.02, 0.022, "Shading: darker = higher R² within that split. Bold: best R² of the split among median, ffill, "
                          "knn and mice (deletion is scored on its own fewer, complete windows; bfill uses future "
                          "values). Test is reported only; no setting was chosen from it.", fontsize=8.5, color=MUTED)
    fig.text(0.02, 0.006, "R² also depends on how much PM varies in that period: calibration (monsoon) has the smallest "
                          "errors but also the smallest spread, so its R² is lower than test's.", fontsize=8.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save_figure(fig, path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def r2_chart(rows, method, cfg, path):
    """R² by forecast horizon on calibration, one line per model, one panel per target (replaces the RMSE chart).

    What: x = horizon (+1 ... +24 h), y = R² of the published forecast; both panels share one R² scale.
    How:  reads the per-horizon rows of this run; 2px lines, ringed markers, legend outside the right panel.
    """
    r = rows[(rows["method"] == method) & (rows["split"] == "calibration")]
    horizons, targets = cfg["horizons"], cfg["targets"]
    x = np.arange(len(horizons))
    lo, hi = r["r2"].min(), r["r2"].max()
    pad = (hi - lo) * 0.12
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.6), facecolor=SURFACE, sharey=True)
    for ax, target in zip(axes, targets):
        for model in cfg["forecasters"]["names"]:
            g = r[(r["target"] == target) & (r["model"] == model)].set_index("horizon").reindex(horizons)
            ax.plot(x, g["r2"], color=COLORS[model], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[model])
        style(ax, target, "R²" if ax is axes[0] else "")
        ax.set_xticks(x, [f"+{h} h" for h in horizons])
        ax.set_xlabel("forecast horizon", color=MUTED)
        ax.set_ylim(lo - pad, min(1, hi + pad))
    axes[-1].legend()                                     # save_figure moves it outside the right panel
    fig.suptitle(f"How much of the real PM variation each model explains (R²): {method} run, calibration May–Aug 2025",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.905, "Higher is better (1 = perfect). Published forecast = median of the quantile model. "
                          "Every split and imputation run: see the metrics table.", fontsize=9, color=SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Build figures_dir/metrics_table.csv, metrics_summary.csv, one table picture per target and each run's R² chart.

    What: prints the R² summary (mean over horizons) and the best R² per split and target.
    How:  metric_rows -> summary_table -> table_chart per target; r2_chart per imputation run.
    """
    cfg = load_config()
    m = cfg["metrics"]
    rows = metric_rows(cfg)
    summary = summary_table(rows)
    rows.round(4).to_csv(cfg["figures_dir"] / m["file"], index=False)
    summary.round(4).to_csv(cfg["figures_dir"] / m["summary_file"], index=False)
    for target in cfg["targets"]:
        table_chart(summary, target, cfg, cfg["figures_dir"] / m["table_plot"].format(target=short(target)))
    for method in cfg["imputation"]["methods"]:
        r2_chart(rows, method, cfg, run_config(method)["figures_dir"] / m["r2_plot"])
    wide = summary.pivot_table(index=["target", "method", "model"], columns="split", values="r2", sort=False)
    best = summary.loc[summary.groupby(["target", "split"], sort=False)["r2"].idxmax(),
                       ["target", "split", "method", "model", "r2"]]
    text = "\n".join(["R² (mean over horizons), published forecast:", wide.round(3).to_string(), "",
                      "best R² per target and split:", best.round(3).to_string(index=False)])
    print(text)
    save_text(cfg, "metrics_table_report.txt", text)


if __name__ == "__main__":
    main()
