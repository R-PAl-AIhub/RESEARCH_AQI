"""Step 6.4 search: which gamma (step size of the adaptive level) gives coverage closest to 1 - alpha?

CALIBRATION only (the test split is never read): odd calendar weeks give the stored CQR scores,
the even (check) weeks are walked in time order with the adaptive level of step 6.4.
Pick rule (chosen 2026-10-27): coverage closest to 1 - alpha, the narrower range breaks a tie
(gaps within worker_c.gamma_tie of the closest count as a tie).
gamma does not change Green / Yellow / Red (they come from Worker B); it only moves the range edges.
With AQI_IMPUTATION set: one run (every model). Without it: every run together (methods that use
future values left out) -> the value for config.yaml worker_c.gamma.
"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.eval.imputation_compare import run_config
from src.models.compare import COLORS, INK, LABELS, MUTED, SURFACE, style
from src.models.train import load, short
from src.plot_style import save_figure
from src.splits.weeks import alternating_weeks
from src.temp_out import save_text
from src.worker_b.distance import group_of
from src.worker_c.methods import adaptive_ranges, stored_scores

METHOD_COLORS = ["#2a78d6", "#eb6834", "#3a9e5c", "#8e44ad", "#d4a017", "#c0392b"]


def gamma_rows(cfg, model):
    """Coverage and average width of the adaptive range on the check weeks, per gamma, target and horizon.

    What: one row per (gamma, target, horizon): coverage (share of real values inside
          [lower, upper]) and width (average upper - lower, µg/m³).
    How:  stored scores = CQR scores of the reference (odd) weeks per group (step 6.3);
          adaptive_ranges walks the check (even) weeks, the level starting at alpha (step 6.4).
    """
    calib = load(model, "calibration")
    part = alternating_weeks(calib["timestamp"]).to_numpy()
    ref, chk = calib[part == "reference"], calib[part == "check"].reset_index(drop=True)
    keys = [f"{short(t)}_h{h}" for t in cfg["targets"] for h in cfg["horizons"]]
    group_by = cfg["worker_c"]["group_by"]
    scores, groups = stored_scores(ref, group_of(ref, group_by), keys), group_of(chk, group_by)
    rows = []
    for gamma in cfg["worker_c"]["gamma_candidates"]:
        for target in cfg["targets"]:
            for h in cfg["horizons"]:
                key = f"{short(target)}_h{h}"
                low, high, _ = adaptive_ranges(chk, groups, scores, key, h, cfg["alpha"], gamma)
                y = chk[f"target_{key}"].to_numpy()
                rows.append({"model": model, "gamma": gamma, "target": target, "horizon": h,
                             "coverage": ((y >= low) & (y <= high)).mean(), "width": (high - low).mean()})
    return rows


def pick(table, aim, tie):
    """The gamma whose coverage is closest to `aim` (the narrower range breaks a tie).

    What: returns (best gamma, table per gamma with coverage, width and gap = |coverage - aim|).
    How:  means over every other column of `table` (models, targets, horizons, methods);
          every gamma whose gap is within `tie` of the smallest gap counts as tied, and the
          tied gamma with the smallest average width wins.
    """
    per = table.groupby("gamma")[["coverage", "width"]].mean()
    per["gap"] = (per["coverage"] - aim).abs()
    tied = per[per["gap"] <= per["gap"].min() + tie]
    return tied["width"].idxmin(), per


def chart(table, line_col, line_names, line_colors, cfg, best, path, title):
    """Coverage (top row) and width (bottom row) against gamma, one panel per horizon, one line per `line_col` value.

    What: dashed line at 1 - alpha; the chosen gamma shaded; each panel title names the
          gamma that is best for that horizon alone.
    How:  means over targets (and models when the lines are methods); gamma on an even
          spaced axis (the values are 0 and then roughly doubling).
    """
    gammas, horizons, aim, tie = (cfg["worker_c"]["gamma_candidates"], cfg["horizons"], 1 - cfg["alpha"],
                                  cfg["worker_c"]["gamma_tie"])
    x = np.arange(len(gammas))
    fig, axes = plt.subplots(2, len(horizons), figsize=(4.2 * len(horizons), 7.5), facecolor=SURFACE, sharex=True)
    for col, h in enumerate(horizons):
        part = table[table["horizon"] == h]
        h_best, _ = pick(part, aim, tie)
        for row, (value, unit) in enumerate([("coverage", "share inside the range"), ("width", "µg/m³")]):
            ax = axes[row, col]
            for name in line_names:
                v = part[part[line_col] == name].groupby("gamma")[value].mean().reindex(gammas)
                ax.plot(x, v * 100 if value == "coverage" else v, marker="o", markersize=3, linewidth=1.4,
                        color=line_colors[name], label=LABELS.get(name, name))
            ax.axvspan(gammas.index(best) - 0.4, gammas.index(best) + 0.4, color=MUTED, alpha=0.15, linewidth=0)
            if value == "coverage":
                ax.axhline(aim * 100, color=INK, linestyle="--", linewidth=1, label=f"aim {aim:.0%}")
                style(ax, f"+{h} h: best gamma {h_best:g}", "% inside the range")
            else:
                style(ax, f"+{h} h: average width", unit)
                ax.set_xticks(x, [f"{g:g}" for g in gammas], rotation=45)
                ax.set_xlabel("gamma", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Run the gamma search for one run (AQI_IMPUTATION set) or for every run together.

    What: saves figures_dir/gamma_compare.csv and .png and prints the best gamma.
          One run: every model of that run. Every run: reads each run's CSV (run it per
          run first); methods in imputation.leaks_future are left out.
    How:  gamma_rows per model; pick() on the mean over everything; test is never read.
    """
    cfg = load_config()
    w, aim = cfg["worker_c"], 1 - cfg["alpha"]
    tie = w["gamma_tie"]
    if cfg.get("run_name"):
        table = pd.DataFrame([r for m in cfg["forecasters"]["names"] for r in gamma_rows(cfg, m)])
        lines, names, colors, where = cfg["forecasters"]["names"], "model", COLORS, f"run {cfg['run_name']}"
    else:
        methods = [m for m in cfg["imputation"]["methods"] if m not in cfg["imputation"]["leaks_future"]]
        table = pd.concat([pd.read_csv(run_config(m)["figures_dir"] / w["gamma_file"]).assign(method=m)
                           for m in methods], ignore_index=True)
        lines, names, colors = methods, "method", dict(zip(methods, METHOD_COLORS))
        where = f"every run ({', '.join(methods)}), mean over the 5 models"
    best, per = pick(table, aim, tie)
    cfg["figures_dir"].mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(cfg["figures_dir"] / w["gamma_file"], index=False)
    chart(table, names, lines, colors, cfg, best, cfg["figures_dir"] / w["gamma_plot_file"],
          f"Step 6.4 - gamma search on the check weeks of calibration, {where}: "
          f"best gamma {best:g} (coverage closest to {aim:.0%}, ties within {tie:.1%} -> narrower)")
    by_h = {h: float(pick(table[table["horizon"] == h], aim, tie)[0]) for h in cfg["horizons"]}
    text = "\n".join([f"gamma search, {where} (calibration check weeks only; aim coverage {aim:.0%}, tie {tie:.1%})",
                      per.round(4).to_string(),
                      f"\nbest per horizon: {by_h}",
                      f"best gamma overall: {best:g}   (config worker_c.gamma now {w['gamma']})"])
    print(text)
    save_text(cfg, "6.4_gamma_search.txt", text)


if __name__ == "__main__":
    main()
