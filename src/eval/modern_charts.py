"""Teacher update (2026-11-02): cleaner charts, one message each (report only, nothing is chosen from them).

Reads the outputs the pipeline already wrote (forecast tables, step 7.1 decisions, step 7.2 tables, the
imputation comparison and the cleaned CPCB data); writes new PNGs next to the old ones (old ones untouched).
Colours: the colourblind-checked model palette (compare.COLORS), the fixed status colours for the trust flag,
one-hue blue for densities.
"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import load_config
from src.eval.imputation_compare import run_config
from src.models.compare import COLORS, INK, LABELS, MUTED, SURFACE, style
from src.models.train import short
from src.plot_style import save_figure

SECONDARY, RULE = "#52514e", "#c3c2b7"
NORMAL, SMOGGY = "#2a78d6", "#eb6834"                                    # categorical slots 1 and 2
FLAGS = {"Green": "#0ca30c", "Yellow": "#fab219", "Red": "#d03b3b"}       # fixed status colours
FLAG_TEXT = {"Green": "#ffffff", "Yellow": "#0b0b0b", "Red": "#ffffff"}
FLAG_NAMES = {"Green": "Green: 90% range", "Yellow": "Yellow: wider range", "Red": "Red: no range, alert only"}
DENSITY = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
METHODS = {"(a) split conformal": "Split conformal", "(b) CQR": "CQR", "(c) CQR + adaptive": "CQR + adaptive (ours)"}


def header(fig, title, note, y_note=0.905):
    """Title (bold, left) and a one-line note under it."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, y_note, note, fontsize=9, color=SECONDARY)


def scatter_chart(test, cfg, path):
    """Forecast vs real value on test as density hexbins: rows = targets, columns = every horizon.

    What: dashed 1:1 line = perfect forecast; each panel names its R² and RMSE; within a target every panel
          has the same axes, cut at the 99.5th percentile of the real values (share beyond written in the panel).
    How:  hexbin with log counts, one-hue blue ramp.
    """
    hs = cfg["horizons"]
    fig, axes = plt.subplots(len(cfg["targets"]), len(hs), figsize=(4.1 * len(hs), 9.2), facecolor=SURFACE)
    for r, target in enumerate(cfg["targets"]):
        top = np.nanpercentile(test[f"target_{short(target)}_h{hs[0]}"].to_numpy(float), 99.5)
        for c, h in enumerate(hs):
            ax, key = axes[r, c], f"{short(target)}_h{h}"
            y, p = test[f"target_{key}"].to_numpy(float), test[f"median_{key}"].to_numpy(float)
            keep = ~(np.isnan(y) | np.isnan(p))
            y, p = y[keep], p[keep]
            beyond = np.mean((y > top) | (p > top)) * 100
            hb = ax.hexbin(y, p, gridsize=45, bins="log", mincnt=1, cmap=DENSITY, extent=(0, top, 0, top),
                           linewidths=0)
            ax.plot([0, top], [0, top], color=INK, linewidth=1, linestyle="--", label="perfect forecast")
            r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
            rmse = np.sqrt(np.mean((p - y) ** 2))
            ax.text(0.05, 0.95, f"R² {r2:.2f}\nRMSE {rmse:.1f}", transform=ax.transAxes, va="top",
                    fontsize=9.5, color=INK)
            ax.text(0.96, 0.04, f"{beyond:.1f}% beyond", transform=ax.transAxes, ha="right", fontsize=7.5,
                    color=MUTED)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            style(ax, f"{target} · +{h} h", "forecast (µg/m³)" if c == 0 else "")
            ax.set_xlabel("real value (µg/m³)", color=MUTED)
    bar = fig.colorbar(hb, ax=axes, fraction=0.012, pad=0.02)
    bar.set_label("number of hours (log scale)", color=MUTED)
    bar.outline.set_visible(False)
    axes[0, -1].legend()
    m = cfg["modern"]
    header(fig, f"Forecast vs real value at every horizon, test months ({LABELS[m['model']]}, {m['run']} run)",
           "Sep–Dec 2025, used only for this report. Closer to the dashed line = better. Darker = more hours. "
           "RMSE in µg/m³. The cloud widens and tilts below the line as the horizon grows.", y_note=0.935)
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.32, right=0.88)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def box_chart(tables, cfg, path):
    """Size of the forecast error per model on test: boxes, rows = targets, columns = every horizon.

    What: box = middle half of the hours, line = median, whiskers = 5th to 95th percentile (no single dots);
          one error scale per target, so the horizons compare directly.
    How:  |forecast - real| per test hour for every model of the run.
    """
    hs = cfg["horizons"]
    names = cfg["forecasters"]["names"]
    fig, axes = plt.subplots(len(cfg["targets"]), len(hs), figsize=(4.0 * len(hs), 8.8), facecolor=SURFACE,
                             sharey="row")
    for r, target in enumerate(cfg["targets"]):
        row_top = 0.0
        for c, h in enumerate(hs):
            ax, key = axes[r, c], f"{short(target)}_h{h}"
            data = [(t[f"median_{key}"] - t[f"target_{key}"]).abs().dropna().to_numpy() for t in
                    (tables[n] for n in names)]
            row_top = max(row_top, max(np.percentile(d, 95) for d in data))
            box = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
                             medianprops={"color": INK, "linewidth": 1.6})
            for patch, name in zip(box["boxes"], names):
                patch.set(facecolor=COLORS[name], alpha=0.35, edgecolor=COLORS[name], linewidth=1.4)
            for part in ("whiskers", "caps"):
                for line, name in zip(box[part], np.repeat(names, 2)):
                    line.set(color=COLORS[name], linewidth=1.4)
            for k, d in enumerate(data, 1):
                ax.text(k, np.median(d), f"{np.median(d):.1f}", va="bottom", ha="center", fontsize=7.5, color=INK)
            ax.set_xticks(range(1, len(names) + 1), [LABELS[n] for n in names], rotation=40, ha="right")
            style(ax, f"{target} · +{h} h", "|forecast − real| (µg/m³)" if c == 0 else "")
        axes[r, 0].set_ylim(0, row_top * 1.05)                # shared by the row: every whisker fits
    m = cfg["modern"]
    header(fig, f"How big the forecast errors are at every horizon, test months ({m['run']} run)",
           "Box = middle half of the hours, line and number = median error, whiskers = 5th to 95th percentile. "
           "Lower = better. Same scale across each row.", y_note=0.935)
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.08)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def imputation_chart(cfg, path):
    """Calibration RMSE of each imputation run on the hours every run kept: dot plot, one dot per model.

    What: rows = imputation runs, x = RMSE (µg/m³, mean over horizons); black bar = mean of the 5 models;
          runs that use future values are marked. Panels: PM2.5, PM10.
    How:  reads figures_dir/imputation_compare.csv (common_rmse_<target> columns).
    """
    t = pd.read_csv(cfg["figures_dir"] / cfg["imputation"]["compare_file"])
    methods, names = cfg["imputation"]["methods"], cfg["forecasters"]["names"]
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(12, 4.8), facecolor=SURFACE)
    for ax, target in zip(axes, cfg["targets"]):
        col = f"common_rmse_{target}"
        order = t.groupby("method")[col].mean().reindex(methods).sort_values(ascending=False).index
        for y, method in enumerate(order):
            g = t[t["method"] == method].set_index("model")[col]
            ax.plot([g.min(), g.max()], [y, y], color=RULE, linewidth=3, solid_capstyle="round", zorder=1)
            for name in names:
                ax.scatter(g[name], y, s=60, color=COLORS[name], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                           label=LABELS[name])
            ax.scatter(g.mean(), y, marker="|", s=320, color=INK, linewidth=2.2, zorder=4, label="mean of 5 models")
        labels = [f"{m} (uses future values)" if m in cfg["imputation"]["leaks_future"] else m for m in order]
        ax.set_yticks(range(len(order)), labels)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        style(ax, target, "")
        ax.set_xlabel("calibration RMSE (µg/m³), same hours for every run", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[-1].legend(keep.values(), keep.keys())
    n = int(t["common_samples"].iloc[0])
    header(fig, "Which way of filling gaps gives the best forecasts?",
           f"Calibration May–Aug 2025, the same {n:,} hours for every run. Left = smaller error. Top row = best mean.",
           y_note=0.885)
    fig.subplots_adjust(top=0.8, wspace=0.55, left=0.16)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def coverage_chart(t1, cfg, path):
    """Coverage on normal vs smoggiest-5% test hours: dumbbells, rows = targets, columns = range methods.

    What: one row per horizon; blue dot = normal hours, orange dot = smoggiest 5%, thin lines = 95% bootstrap
          band; dashed line = the 90% aim.
    How:  reads the step 7.2 table 1 of the explained model.
    """
    fig, axes = plt.subplots(len(cfg["targets"]), len(METHODS), figsize=(14, 7.5), facecolor=SURFACE, sharex=True)
    hs = cfg["horizons"]
    for r, target in enumerate(cfg["targets"]):
        for c, (method, name) in enumerate(METHODS.items()):
            ax = axes[r, c]
            g = t1[(t1["target"] == target) & (t1["method"] == method)]
            for y, h in enumerate(hs):
                n, e = (g[(g["horizon"] == h) & (g["group"] == grp)].iloc[0] for grp in ("normal", "extreme"))
                ax.plot([e["coverage"] * 100, n["coverage"] * 100], [y, y], color=RULE, linewidth=2.5, zorder=1)
                for row, color, label in ((n, NORMAL, "normal hours"), (e, SMOGGY, "smoggiest 5% of hours")):
                    ax.plot([row["coverage 2.5%"] * 100, row["coverage 97.5%"] * 100], [y, y], color=color,
                            linewidth=1, alpha=0.6, zorder=2)
                    ax.scatter(row["coverage"] * 100, y, s=70, color=color, edgecolor=SURFACE, linewidth=1.3,
                               zorder=3, label=label)
            ax.axvline((1 - cfg["alpha"]) * 100, color=INK, linewidth=1, linestyle="--", label="aim 90%")
            ax.set_yticks(range(len(hs)), [f"+{h} h" for h in hs])
            ax.set_xlim(-3, 103)
            ax.tick_params(axis="y", length=0, labelcolor=INK)
            style(ax, f"{name} · {target}", "")
            if r == len(cfg["targets"]) - 1:
                ax.set_xlabel("% of real values inside the range", color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[0, -1].legend(keep.values(), keep.keys())
    m = cfg["modern"]
    header(fig, f"Does the range keep its 90% promise? Normal vs smoggiest hours ({LABELS[m['model']]}, test)",
           "Sep–Dec 2025, used only for this report. Dots on the dashed line = promise kept. Thin lines = 95% "
           "bootstrap band (1,000 day blocks).", y_note=0.925)
    fig.subplots_adjust(top=0.86, hspace=0.35, wspace=0.12)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def station_chart(t3, cfg, path):
    """Coverage of the published range (CQR + adaptive) per station on test: dot = mean over horizons, bar = range.

    What: rows = stations sorted by mean coverage; grey bar = lowest to highest horizon; number = mean.
    How:  reads the step 7.2 table 3 of the explained model.
    """
    g = t3[t3["method"] == "(c) CQR + adaptive"]
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(12, 4.8), facecolor=SURFACE, sharex=True)
    for ax, target in zip(axes, cfg["targets"]):
        s = g[g["target"] == target].groupby("group")["coverage"].agg(["mean", "min", "max"]).sort_values("mean") * 100
        for y, (station, row) in enumerate(s.iterrows()):
            ax.plot([row["min"], row["max"]], [y, y], color=RULE, linewidth=5, solid_capstyle="round", zorder=1)
            ax.scatter(row["mean"], y, s=70, color=NORMAL, edgecolor=SURFACE, linewidth=1.3, zorder=3,
                       label="mean over the 5 horizons")
            ax.text(row["max"] + 1.2, y, f"{row['mean']:.0f}%", va="center", fontsize=9, color=INK)
        ax.axvline((1 - cfg["alpha"]) * 100, color=INK, linewidth=1, linestyle="--", label="aim 90%")
        ax.set_xlim(g["coverage"].min() * 100 - 2, 103)          # room for the numbers inside the panel
        ax.set_yticks(range(len(s)), [st.split(",")[0] for st in s.index])
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        style(ax, target, "")
        ax.set_xlabel("% of real values inside the published range", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[-1].legend(keep.values(), keep.keys())
    m = cfg["modern"]
    header(fig, f"Does the 90% promise hold at every station? ({LABELS[m['model']]}, CQR + adaptive, test)",
           "Grey bar = lowest to highest of the 5 horizons, dot and number = their mean. Below the dashed line = "
           "range too narrow there.", y_note=0.885)
    fig.subplots_adjust(top=0.8, wspace=0.45, left=0.14)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def flags_chart(cfg, rc, path):
    """Share of Green / Yellow / Red test hours per model: 100% horizontal bars with the share written inside.

    What: one bar per model; the flag is the same for every horizon of an hour (step 5.4), so flag_h1 is used.
    How:  reads each model's step 7.1 decisions_test.csv.
    """
    names = cfg["forecasters"]["names"]
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    for y, name in enumerate(names[::-1]):
        flag = pd.read_csv(rc["forecasts_dir"] / name / cfg["evaluation"]["flags_file"], usecols=["flag_h1"])["flag_h1"]
        left = 0.0
        for f, color in FLAGS.items():
            share = (flag == f).mean() * 100
            ax.barh(y, share, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2,
                    label=FLAG_NAMES[f])
            if share >= 4:
                ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center", fontsize=9,
                        color=FLAG_TEXT[f], fontweight="bold")
            left += share
        ax.text(101.5, y, f"Red {share:.1f}%", va="center", fontsize=9, color=INK)   # share = Red, the last flag
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names[::-1]])
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    style(ax, "", "")
    ax.set_xlabel("% of test hours", color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    ax.legend(keep.values(), keep.keys())
    header(fig, f"How often each trust flag is shown on the test months ({cfg['modern']['run']} run)",
           "Sep–Dec 2025. Green = publish the 90% range, Yellow = publish a wider range, Red = abstain "
           "(Red share written at the right).", y_note=0.84)
    fig.subplots_adjust(top=0.72, left=0.12)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def week_chart(test, cfg, path):
    """A few smoggy test days at one station: real value, forecast and the model's 90% band, every horizon.

    What: x = the hour the forecast is for (sample time + horizon); rows = targets, columns = horizons;
          one µg/m³ scale per target.
    How:  quantile table of the explained model (median, low, high); dots = real values.
    """
    m = cfg["modern"]
    start = pd.Timestamp(m["week_start"])
    end = start + pd.Timedelta(days=m["week_days"])
    t = test[test["station"] == m["week_station"]]
    hs = cfg["horizons"]
    fig, axes = plt.subplots(len(cfg["targets"]), len(hs), figsize=(4.2 * len(hs), 7.8), facecolor=SURFACE,
                             sharex=True, sharey="row")
    for r, target in enumerate(cfg["targets"]):
        for c, h in enumerate(hs):
            ax, key = axes[r, c], f"{short(target)}_h{h}"
            w = t.assign(when=t["timestamp"] + pd.Timedelta(hours=h)).set_index("when").sort_index().loc[start:end]
            ax.fill_between(w.index, w[f"low_{key}"], w[f"high_{key}"], color="#b7d3f6", linewidth=0,
                            label="model's 90% band")
            ax.plot(w.index, w[f"median_{key}"], color=NORMAL, linewidth=2, label="forecast")
            ax.scatter(w.index, w[f"target_{key}"], s=4, color=INK, zorder=3, label="real value")
            style(ax, f"{target} · +{h} h", "µg/m³" if c == 0 else "")
            ax.set_ylim(bottom=0)
    axes[0, -1].legend()
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    header(fig, f"Forecast and band over {m['week_days']} smoggy test days at {m['week_station'].split(',')[0]} "
                f"({LABELS[m['model']]}, {m['run']} run)",
           f"From {start:%d %b %Y}. Black dots = real values, blue line = forecast, shaded = the model's own 90% band "
           "(before the conformal correction).", y_note=0.925)
    fig.subplots_adjust(top=0.86, hspace=0.3, wspace=0.08)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def availability_chart(cfg, path):
    """Share of hours with a PM2.5 value, per station and month: heatmap with the value written in each cell.

    What: one-hue blue (darker = more complete); cells under 90% have their value in bold.
    How:  reads the cleaned CPCB table (step 2.2) and counts non-empty PM2.5 hours per station and month.
    """
    d = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_clean_file"], usecols=["station", "time", "PM2.5"],
                    parse_dates=["time"])
    share = d.assign(month=d["time"].dt.to_period("M")).groupby(["station", "month"])["PM2.5"] \
        .apply(lambda v: v.notna().mean() * 100).unstack()
    fig, ax = plt.subplots(figsize=(16, 4.6), facecolor=SURFACE)
    im = ax.imshow(share.to_numpy(), cmap=DENSITY, vmin=0, vmax=100, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v > 60 else INK, fontweight="bold" if v < 90 else "normal")
    ax.set_yticks(range(share.shape[0]), [s.split(",")[0] for s in share.index])
    ax.set_xticks(range(share.shape[1]), [p.strftime("%b\n%Y") if p.month == 1 or j == 0 else p.strftime("%b")
                                          for j, p in enumerate(share.columns)], fontsize=7)
    ax.tick_params(length=0, labelcolor=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label("% of hours with a value", color=MUTED)
    bar.outline.set_visible(False)
    header(fig, "How complete the PM2.5 data is, station by station and month by month",
           "Raw CPCB hourly files 2023–2025 after the quality checks. Bold = under 90% of the hours.", y_note=0.9)
    fig.subplots_adjust(top=0.82, left=0.12)
    save_figure(fig, path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Draw every chart of config modern for every imputation run and model (two of them once, into figures_dir).

    What: per run, the error boxes and flag shares (all models); per run x model, the scatter, coverage
          dumbbells, station coverage and smoggy-days charts in figures_dir/runs/<run>/<model>/; once, the
          imputation dot plot and the availability heatmap.
    How:  sets cfg modern.run / modern.model before each chart (the titles read them); reads the files the
          pipeline already wrote.
    """
    cfg = load_config()
    m = cfg["modern"]
    count = 0
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        out = rc["figures_dir"]
        m["run"] = method
        tables = {n: pd.read_csv(rc["forecasts_dir"] / n / "quantile" / "test.csv", parse_dates=["timestamp"])
                  for n in cfg["forecasters"]["names"]}
        box_chart(tables, cfg, out / m["box_plot"])
        flags_chart(cfg, rc, out / m["flags_plot"])
        count += 2
        for model in cfg["forecasters"]["names"]:
            m["model"] = model
            d = out / model
            d.mkdir(parents=True, exist_ok=True)
            scatter_chart(tables[model], cfg, d / m["scatter_plot"])
            coverage_chart(pd.read_csv(d / cfg["evaluation"]["table1_file"]), cfg, d / m["coverage_plot"])
            station_chart(pd.read_csv(d / cfg["evaluation"]["table3_file"]), cfg, d / m["station_plot"])
            week_chart(tables[model], cfg, d / m["week_plot"])
            count += 4
        print(f"{method}: done", flush=True)
    imputation_chart(cfg, cfg["figures_dir"] / m["imputation_plot"])
    availability_chart(cfg, cfg["figures_dir"] / m["availability_plot"])
    print(f"{count + 2} charts saved")


if __name__ == "__main__":
    main()
