"""Teacher update (2026-11-02): one-look heatmaps of every imputation run x model x horizon (report only).

R² and RMSE of the published forecast (from the metrics table, every split) and test coverage of split
conformal vs CQR + adaptive on normal and smoggiest-5% hours (from each run's step 7.2 table 1).
"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import load_config
from src.eval.imputation_compare import run_config
from src.models.compare import INK, LABELS, MUTED, SURFACE
from src.models.train import short
from src.plot_style import save_figure

SECONDARY = "#52514e"
BLUE = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
GAP = LinearSegmentedColormap.from_list("gap", ["#e34948", "#f0efec", "#2a78d6"])   # too narrow - 90% - too wide
SPLITS = {"train": "Train 2023–24", "validation": "Validation Jan–Apr 2025", "calibration": "Calibration May–Aug 2025",
          "test": "Test Sep–Dec 2025"}
METHODS = {"(a) split conformal": "split conformal", "(c) CQR + adaptive": "CQR + adaptive (ours)"}
GROUPS = {"normal": "normal hours", "extreme": "smoggiest 5%"}


def grid_chart(values, blocks, cfg, title, notes, path, fmt, shade):
    """Heatmap: rows = imputation run x model, column blocks = `blocks` (key -> label) x every horizon.

    What: the value is written in every cell; shade(block, array) -> (array to colour, cmap, vmin, vmax).
    How:  values = DataFrame indexed by (method, model), columns (block, horizon).
    """
    methods, models, hs = cfg["imputation"]["methods"], cfg["forecasters"]["names"], cfg["horizons"]
    rows = [(m, n) for m in methods for n in models]
    fig, ax = plt.subplots(figsize=(3 + 0.55 * len(blocks) * len(hs), 0.3 * len(rows) + 3), facecolor=SURFACE)
    for b, key in enumerate(blocks):
        cols = [(key, h) for h in hs]
        block = values.reindex(index=pd.MultiIndex.from_tuples(rows), columns=pd.MultiIndex.from_tuples(cols)).to_numpy()
        colour, cmap, lo, hi = shade(key, block)
        x0 = b * (len(hs) + 0.6)
        ax.imshow(colour, cmap=cmap, vmin=lo, vmax=hi, aspect="auto",
                  extent=(x0 - 0.5, x0 + len(hs) - 0.5, len(rows) - 0.5, -0.5))
        for i in range(len(rows)):
            for j in range(len(hs)):
                v, c = block[i, j], colour[i, j]
                if not np.isnan(v):
                    dark = abs(c) > 0.7 * hi if cmap is GAP else (c - lo) / (hi - lo) > 0.8
                    ax.text(x0 + j, i, fmt(v), ha="center", va="center", fontsize=7,
                            color="#ffffff" if dark else INK)
        ax.text(x0 + (len(hs) - 1) / 2, -2.2, blocks[key], ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="bold")
        for j, h in enumerate(hs):
            ax.text(x0 + j, -1.0, f"+{h} h", ha="center", va="bottom", fontsize=8, color=SECONDARY)
    ax.set_xlim(-0.5, len(blocks) * (len(hs) + 0.6) - 1.1)
    ax.set_ylim(len(rows) - 0.5, -3.6)
    ax.set_yticks(range(len(rows)), [LABELS[n] for _, n in rows], fontsize=8)
    ax.set_xticks([])
    for k, m in enumerate(methods):
        ax.annotate(m, xy=(0, k * len(models) + (len(models) - 1) / 2), xycoords=("axes fraction", "data"),
                    xytext=(-72, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK,
                    fontweight="bold")
        if k:
            ax.axhline(k * len(models) - 0.5, color=SURFACE, linewidth=3)
    ax.tick_params(length=0, labelcolor=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    for k, line in enumerate(notes):
        fig.text(0.01, 0.03 - 0.018 * k, line, fontsize=8.5, color=SECONDARY if k == 0 else MUTED)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.07)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    """R², RMSE and coverage heatmaps of every run x model x horizon, one picture per target and measure."""
    cfg = load_config()
    a = cfg["all_charts"]
    metrics = pd.read_csv(cfg["figures_dir"] / cfg["metrics"]["file"])
    cov = []
    for method in cfg["imputation"]["methods"]:
        for model in cfg["forecasters"]["names"]:
            t = pd.read_csv(run_config(method)["figures_dir"] / model / cfg["evaluation"]["table1_file"])
            cov.append(t.assign(run=method, model=model))
    cov = pd.concat(cov)
    aim, span = (1 - cfg["alpha"]) * 100, cfg["coverage"]["shade_points"]
    for target in cfg["targets"]:
        m = metrics[metrics["target"] == target]
        r2 = m.pivot_table(index=["method", "model"], columns=["split", "horizon"], values="r2")
        lo, hi = np.nanmin(r2.to_numpy()), np.nanmax(r2.to_numpy())
        grid_chart(r2, SPLITS, cfg, f"R² of the published forecast, every run, model and horizon: {target}",
                   ["Cell = R² (share of the real variation the forecast explains; 1 = perfect) of the median of the "
                    "quantile model. Darker = higher. One colour scale for the whole picture.",
                    "R² is lower in calibration partly because PM varies about 4× less in the monsoon months. Test is "
                    "reported only. Deletion is scored on its own (fewer, complete) windows."],
                   cfg["figures_dir"] / a["r2_plot"].format(target=short(target)), lambda v: f"{v:.2f}",
                   lambda key, block: (block, BLUE, lo, hi))
        rmse = m.pivot_table(index=["method", "model"], columns=["split", "horizon"], values="rmse")
        grid_chart(rmse, SPLITS, cfg, f"RMSE of the published forecast, every run, model and horizon: {target} (µg/m³)",
                   ["Cell = RMSE in µg/m³ (lower is better). Shading is within each split (darker = larger error), "
                    "because test errors are about 3× calibration errors.",
                    "Test is reported only. Deletion is scored on its own (fewer, complete) windows, so its rows are "
                    "not directly comparable; the fair comparison is the common-hours chart."],
                   cfg["figures_dir"] / a["rmse_plot"].format(target=short(target)), lambda v: f"{v:.1f}",
                   lambda key, block: (block, BLUE, np.nanmin(block), np.nanmax(block)))
        c = cov[(cov["target"] == target) & cov["method"].isin(METHODS)]
        c = c.assign(block=c["method"].map(METHODS) + " · " + c["group"].map(GROUPS))
        piv = c.pivot_table(index=["run", "model"], columns=["block", "horizon"], values="coverage") * 100
        piv.index.names = ["method", "model"]
        blocks = {f"{mv} · {gv}": f"{mv}\n{gv}" for mv in METHODS.values() for gv in GROUPS.values()}
        grid_chart(piv, blocks, cfg, f"Test coverage, every run, model and horizon: {target} (aim 90%)",
                   ["Cell = % of real values inside the range on test (Sep–Dec 2025). Grey = 90%, red = too narrow, "
                    f"blue = wider than needed (colour clipped at ±{span} points).",
                    "Normal hours = all but the smoggiest 5% of test hours. Test is reported only; nothing was chosen "
                    "from it."],
                   cfg["figures_dir"] / a["coverage_plot"].format(target=short(target)), lambda v: f"{v:.0f}",
                   lambda key, block: (np.clip(block - aim, -span, span), GAP, -span, span))
        print("saved", target, flush=True)


if __name__ == "__main__":
    main()
