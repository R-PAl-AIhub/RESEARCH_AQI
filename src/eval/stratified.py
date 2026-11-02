"""Step 7.2 (Stage 6, item 1) for PROJECT_2: coverage on normal vs the smoggiest 5% of test hours,
for three methods, every forecaster and both targets, with day-block bootstrap intervals.

PROJECT_2 uses one group for all hours, so the per-group Mondrian methods of PROJECT are not repeated.
"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure
from src.models.compare import COLORS, INK, LABELS, MUTED, SURFACE, style
from src.models.train import load, short
from src.temp_out import save_text
from src.worker_c.methods import adaptive_ranges, cqr_q, split_width, stored_scores

METHODS = {"(a) split conformal": "a split", "(b) CQR": "b CQR", "(c) CQR + adaptive": "c CQR+adapt"}
GROUP_COLORS = {"normal": "#2a78d6", "extreme": "#eb6834"}


def method_ranges(calib, test, key, h, alpha, gamma):
    """Low / high of every method on the test samples.

    What: returns ranges: ranges[method] = (low, high) arrays.
    How:  (a) median +/- w and (b) [low - Q, high + Q] from all calibration;
          (c) the step 6.3 adaptive update on test with one group.
    """
    n = len(test)
    w, q = split_width(calib, key, alpha), cqr_q(calib, key, alpha)
    med, low, high = (test[f"{c}_{key}"].to_numpy() for c in ("median", "low", "high"))
    ranges = {"(a) split conformal": (med - w, med + w), "(b) CQR": (low - q, high + q)}
    a_low, a_high, _ = adaptive_ranges(test, np.zeros(n, int), stored_scores(calib, np.zeros(len(calib), int), [key]),
                                       key, h, alpha, gamma)
    ranges["(c) CQR + adaptive"] = (a_low, a_high)
    return ranges


def day_bootstrap(days, hits, widths, repeats, seed):
    """95% intervals of coverage and average width, resampling whole days.

    What: returns (coverage low, coverage high, width low, width high).
    How:  per day: number of samples, hits and summed width (all stations of
          that day together). Each repeat draws as many days as there are, with
          replacement, and recomputes coverage = hits / samples and width =
          summed width / samples from the day totals. Percentiles 2.5 and 97.5.
    """
    codes, _ = pd.factorize(days)
    n, k, s = (np.bincount(codes, weights=x) for x in (np.ones(len(codes)), hits, widths))
    idx = np.random.default_rng(seed).integers(0, len(n), size=(repeats, len(n)))
    total = n[idx].sum(axis=1)
    cov, wid = k[idx].sum(axis=1) / total, s[idx].sum(axis=1) / total
    return (*np.percentile(cov, [2.5, 97.5]), *np.percentile(wid, [2.5, 97.5]))


def strat_rows(test, ranges, groups, key, target, h, repeats, seed):
    """Per group and method: samples, days, coverage, width, 95% intervals.

    What: list of dict rows for one target and horizon. groups = group name of
          every test sample.
    How:  hit = low <= actual <= high; days = distinct calendar days of the
          samples counted; intervals from day_bootstrap.
    """
    rows = []
    days_all = test["timestamp"].dt.normalize().to_numpy()
    y = test[f"target_{key}"].to_numpy()
    for g in pd.unique(groups):
        in_g = groups == g
        for method, (low, high) in ranges.items():
            m = in_g
            hits, widths = ((low <= y) & (y <= high))[m], (high - low)[m]
            c_lo, c_hi, w_lo, w_hi = day_bootstrap(days_all[m], hits, widths, repeats, seed)
            rows.append({"target": target, "horizon": h, "group": g, "method": method, "samples": int(m.sum()),
                         "days": len(np.unique(days_all[m])), "coverage": hits.mean(),
                         "coverage 2.5%": c_lo, "coverage 97.5%": c_hi, "width": widths.mean(),
                         "width 2.5%": w_lo, "width 97.5%": w_hi})
    return rows


def coverage_chart(table, model, target, alpha, horizons, out):
    """One forecaster, one target: normal vs extreme coverage of every method, one panel per horizon.

    What: grouped bars (blue = normal, orange = extreme) with 95% error bars,
          sample counts under the bars and the aim line.
    How:  values from Table 1.
    """
    cols = -(-len(horizons) // 2)                     # 2 rows, enough columns for every horizon
    fig, axes = plt.subplots(2, cols, figsize=(8 * cols, 10), facecolor=SURFACE, sharey=True)
    for ax in axes.ravel()[len(horizons):]:
        ax.axis("off")
    for ax, h in zip(axes.ravel(), horizons):
        t = table[(table["horizon"] == h) & (table["target"] == target)]
        methods = list(METHODS)
        x, width = np.arange(len(methods)), 0.38
        for k, group in enumerate(["normal", "extreme"]):
            g = t[t["group"] == group].set_index("method").loc[methods]
            err = np.vstack([g["coverage"] - g["coverage 2.5%"], g["coverage 97.5%"] - g["coverage"]]) * 100
            pos = x + (k - 0.5) * (width + 0.03)
            bars = ax.bar(pos, g["coverage"] * 100, width, color=GROUP_COLORS[group], yerr=err, capsize=3,
                          ecolor=INK, label=f"{group} hours")
            ax.bar_label(bars, fmt="%.0f", padding=8, fontsize=7, color=INK)
            for p, n in zip(pos, g["samples"]):
                ax.text(p, 2, f"n={n}", rotation=90, ha="center", va="bottom", fontsize=6, color=SURFACE)
        ax.axhline((1 - alpha) * 100, color=INK, linestyle="--", linewidth=1, label=f"aim {1 - alpha:.0%}")
        ax.set_xticks(x, [METHODS[m] for m in methods], rotation=25, ha="right")
        ax.set_ylim(0, 108)
        style(ax, f"+{h} h: share of real {target} inside the range (test Sep–Dec 2025)", "% of samples")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=10, labelcolor=INK, loc="upper right", ncol=3)
    fig.suptitle(f"{LABELS[model]} - {target}: normal vs smoggiest 5% hours, coverage with 95% day-block "
                 f"bootstrap intervals", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def compare_chart(tables, target, alpha, horizons, out):
    """All forecasters side by side for one target: rows = normal / extreme, columns = horizons.

    What: in each panel, one bar group per method with one bar per forecaster,
          95% error bars and the aim line, so models and methods can be compared at once.
    How:  values from every model's Table 1.
    """
    models, methods = list(tables), list(METHODS)
    fig, axes = plt.subplots(2, len(horizons), figsize=(6 * len(horizons), 10), facecolor=SURFACE, sharey=True)
    width = 0.8 / len(models)
    x = np.arange(len(methods))
    for r, group in enumerate(["normal", "extreme"]):
        for c, h in enumerate(horizons):
            ax = axes[r, c]
            for k, model in enumerate(models):
                t = tables[model]
                g = t[(t["target"] == target) & (t["horizon"] == h) & (t["group"] == group)].set_index("method").loc[methods]
                err = np.vstack([g["coverage"] - g["coverage 2.5%"], g["coverage 97.5%"] - g["coverage"]]) * 100
                ax.bar(x + (k - (len(models) - 1) / 2) * width, g["coverage"] * 100, width * 0.92,
                       color=COLORS[model], yerr=err, capsize=1.5, ecolor=MUTED, label=LABELS[model])
            ax.axhline((1 - alpha) * 100, color=INK, linestyle="--", linewidth=1)
            ax.set_xticks(x, [METHODS[m] for m in methods], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            style(ax, f"{group} hours, +{h} h", "% inside the range" if c == 0 else "")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle(f"{target}: coverage of every method and forecaster, test Sep–Dec 2025 (dashed = aim {1 - alpha:.0%})",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out, dpi=120, facecolor=SURFACE)
    plt.close(fig)


METHOD_COLORS = ["#b7b5ae", "#6aa6e8", "#2a78d6", "#8e44ad", "#3a9d5d", "#c0392b"]


def station_chart(table3, model, target, alpha, horizons, out):
    """One forecaster, one target: coverage of every method per station (test Sep–Dec 2025), one panel per horizon.

    What: grouped bars, x = station, one bar per method, dashed aim line.
    How:  values from Table 3 (groups = stations, all test hours of that station).
    """
    fig, axes = plt.subplots(len(horizons), 1, figsize=(16, 3.4 * len(horizons)), facecolor=SURFACE, sharex=True)
    t = table3[table3["target"] == target]
    stations = sorted(t["group"].unique())
    x, methods = np.arange(len(stations)), list(METHODS)
    width = 0.8 / len(methods)
    for ax, h in zip(axes, horizons):
        e = t[t["horizon"] == h].pivot(index="group", columns="method", values="coverage").loc[stations]
        for k, (method, color) in enumerate(zip(methods, METHOD_COLORS)):
            ax.bar(x + (k - (len(methods) - 1) / 2) * width, e[method] * 100, width * 0.92, color=color,
                   label=METHODS[method])
        ax.axhline((1 - alpha) * 100, color=INK, linestyle="--", linewidth=1)
        ax.set_ylim(40, 102)
        style(ax, f"+{h} h", "% inside the range")
    axes[-1].set_xticks(x, [s.replace(", Kolkata - WBPCB", "") for s in stations], rotation=15)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=6, loc="lower left")
    fig.suptitle(f"{LABELS[model]} - {target}: coverage per station and method, test Sep–Dec 2025 (dashed = aim)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def compact(table):
    """Short printable view: "coverage [interval] w width" per (target, horizon, method) and group."""
    cell = (table["coverage"].mul(100).map("{:.1f}".format) + " [" + table["coverage 2.5%"].mul(100).map("{:.1f}".format)
            + "-" + table["coverage 97.5%"].mul(100).map("{:.1f}".format) + "] w " + table["width"].map("{:.0f}".format))
    return table.assign(cell=cell, method=table["method"].map(METHODS)).pivot_table(
        index=["target", "horizon", "method"], columns="group", values="cell", aggfunc="first", sort=False)


def main():
    """Run step 7.2 on test for every forecaster: Table 1 (normal / extreme), Table 2 (seasons), charts.

    What: per model saves figures/<model>/stratified_table1.csv, stratified_table2.csv
          and one coverage chart per target; then one comparison chart per target
          with all models. The extreme rule was fixed in config.yaml before this ran.
    How:  extreme = target >= the (1 - extreme_top_share) quantile of that target
          and horizon's observed value in test.
    """
    cfg = load_config()
    e_cfg, targets, horizons, alpha = cfg["evaluation"], cfg["targets"], cfg["horizons"], cfg["alpha"]
    gamma = cfg["worker_c"]["gamma"]
    tables1, lines = {}, [f"alpha {alpha}, extreme = top {e_cfg['extreme_top_share']:.0%} of the observed value "
                          f"of each target and horizon in test ({cfg['splits']['test_start']} to "
                          f"{cfg['splits']['test_end']}); Worker C groups = {cfg['worker_c']['group_by']}"]
    for model in cfg["forecasters"]["names"]:
        calib, test = load(model, "calibration"), load(model, "test")
        rows1, rows2, rows3, cuts = [], [], [], {}
        for target in targets:
            for h in horizons:
                key = f"{short(target)}_h{h}"
                ranges = method_ranges(calib, test, key, h, alpha, gamma)
                cuts[key] = test[f"target_{key}"].quantile(1 - e_cfg["extreme_top_share"])
                extreme = np.where(test[f"target_{key}"].to_numpy() >= cuts[key], "extreme", "normal")
                rows1 += strat_rows(test, ranges, extreme, key, target, h,
                                    e_cfg["bootstrap_repeats"], cfg["seed"])
                rows2 += strat_rows(test, ranges, test["season"].to_numpy(), key, target, h,
                                    e_cfg["bootstrap_repeats"], cfg["seed"])
                rows3 += strat_rows(test, ranges, test["station"].to_numpy(), key, target, h,
                                    e_cfg["bootstrap_repeats"], cfg["seed"])
        table1, table2, table3 = pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(rows3)
        out_dir = cfg["figures_dir"] / model
        out_dir.mkdir(parents=True, exist_ok=True)
        table1.round(4).to_csv(out_dir / e_cfg["table1_file"], index=False)
        table2.round(4).to_csv(out_dir / e_cfg["table2_file"], index=False)
        table3.round(4).to_csv(out_dir / e_cfg["table3_file"], index=False)
        for target in targets:
            coverage_chart(table1, model, target, alpha, horizons,
                           out_dir / e_cfg["coverage_plot_file"].format(target=short(target)))
            station_chart(table3, model, target, alpha, horizons,
                          out_dir / e_cfg["station_plot_file"].format(target=short(target)))
        tables1[model] = table1
        lines += [f"\n===== {model} =====  extreme cut (µg/m³): " + ", ".join(f"{k} >= {v:.0f}" for k, v in cuts.items()),
                  compact(table1)[["normal", "extreme"]].to_string()]
    for target in targets:
        compare_chart(tables1, target, alpha, horizons,
                      cfg["figures_dir"] / e_cfg["compare_plot_file"].format(target=short(target)))
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "7.2_stratified_report.txt", text)


if __name__ == "__main__":
    main()
