"""Part 5 charts: one file per forecaster (figures/<model>/) and comparison files (figures/), calibration only."""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure
from src.models.compare import COLORS, INK, LABELS, MUTED, SURFACE, style
from src.worker_b.distance import season_order, worker_dir

FLAG_COLORS = {"Green": "#3a9d5d", "Yellow": "#e3b505", "Red": "#c0392b"}
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#8a8f98", "calibration": "#eb6834"}
PERIOD_COLORS = {"train": "#2a78d6", "validation": "#8a8f98", "calibration": "#eb6834", "test": "#3a9d5d"}
PERIOD_NAMES = {"train": "train 2023-24", "validation": "validation Jan-Apr 2025",
                "calibration": "calibration May-Aug 2025", "test": "test Sep-Dec 2025"}   # option C


def save(fig, path, title):
    """Add the title, lay out, save at 130 dpi and close."""
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def distance_chart(dist, order, model, path):
    """5.2, one model: histogram of the distance to the own season, train vs calibration, one panel per season.

    What: two overlaid histograms per season, with the train 99th percentile marked.
    How:  same bins for both splits; densities so different sizes compare.
    """
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4), facecolor=SURFACE, sharey=True)
    top = dist["distance"].quantile(0.995)
    bins = np.linspace(0, top, 60)
    for ax, season in zip(axes, order):
        for split, color in SPLIT_COLORS.items():
            d = dist.loc[(dist["season"] == season) & (dist["split"] == split), "distance"]
            ax.hist(d.clip(upper=top), bins=bins, density=True, alpha=0.55, color=color,
                    label=f"{split} (median {d.median():.1f})")
        p99 = dist.loc[(dist["season"] == season) & (dist["split"] == "train"), "distance"].quantile(0.99)
        ax.axvline(p99, color=INK, linestyle="--", linewidth=1)
        ax.annotate("train p99", (p99, ax.get_ylim()[1] * 0.9), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        ax.set_xlabel("distance to own season (Mahalanobis)", color=MUTED)
        style(ax, season, "density")
    save(fig, path, f"5.2 - {LABELS[model]}: distance of each sample to its season's shape (embedding)")


def raw_chart(check, model, path):
    """5.6, one model: s_raw vs s_embed on the check weeks (density), and which score is larger.

    What: left = hexbin of the two scores with the correlation in the title;
          right = share of samples where s_raw, s_embed or neither is larger.
    How:  reads worker_b/raw_check.csv.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.4, 1]})
    r = np.corrcoef(check["s_raw"], check["s_embed"])[0, 1]
    hb = axes[0].hexbin(check["s_embed"], check["s_raw"], gridsize=40, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=axes[0], label="samples (log)")
    axes[0].plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1)
    axes[0].set_xlabel("s_embed (model's notepad)", color=MUTED)
    style(axes[0], f"correlation r = {r:.2f} (related, not identical is good)", "s_raw (raw inputs)")
    shares = pd.Series({"s_raw larger": (check["s_raw"] > check["s_embed"]).mean(),
                        "s_embed larger": (check["s_embed"] > check["s_raw"]).mean(),
                        "equal": (check["s_raw"] == check["s_embed"]).mean()}) * 100
    bars = axes[1].bar(shares.index, shares, color=["#2a78d6", "#eb6834", MUTED])
    axes[1].bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, color=INK)
    style(axes[1], "which view finds the hour more unusual", "% of check-week samples")
    save(fig, path, f"5.6 - {LABELS[model]}: raw-feature safety net vs embedding score (check weeks of calibration May-Aug 2025)")


def flag_rules(w):
    """Legend text per colour with its band of the fused score s, e.g. "Green (s ≤ 0.80)"."""
    g, y = w["green_max"], w["yellow_max"]
    return {"Green": f"Green (s ≤ {g:.2f})", "Yellow": f"Yellow ({g:.2f} < s ≤ {y:.2f})", "Red": f"Red (s > {y:.2f})"}


def labels_chart(shares, order, horizons, model, path, rules):
    """5.6b, one model: stacked Green / Yellow / Red shares per season, one panel per horizon.

    What: stacked bars with the Red share written on top; dashed lines at 80% and 98%.
    How:  reads worker_b/label_check.csv (check weeks ranked against reference weeks).
    """
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.8), facecolor=SURFACE, sharey=True)
    groups = ["all"] + [s for s in order if s in set(shares["season"])]   # seasons present
    for ax, h in zip(axes, horizons):
        s = shares[shares["horizon"] == h].set_index("season").loc[groups]
        bottom = np.zeros(len(groups))
        for flag, color in FLAG_COLORS.items():
            ax.bar(groups, s[flag] * 100, bottom=bottom, color=color, label=rules[flag])
            bottom += s[flag].to_numpy() * 100
        for i, red in enumerate(s["Red"]):
            ax.text(i, 101, f"R {red * 100:.1f}", ha="center", fontsize=7, color=INK)
        for lv in (80, 98):
            ax.axhline(lv, color=INK, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 106)
        ax.tick_params(axis="x", labelrotation=30)
        style(ax, f"+{h} h", "% of calibration week samples")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    save(fig, path, f"5.6b - {LABELS[model]}: label shares on calibration weeks (from the fused score)")


def timeline_chart(labels, station, horizons, model, path):
    """5.6b, one model: fused score of one station through calibration, coloured by label.

    What: dots coloured Green / Yellow / Red; dashed lines at the cut-points. One panel:
          with no smoothing the score and label are the same for every horizon.
    How:  reads worker_b/worker_b_calibration.csv (ranked against all of calibration).
    """
    s = labels[labels["station"] == station].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(18, 3), facecolor=SURFACE)
    for flag, color in FLAG_COLORS.items():
        m = s[f"label_h{horizons[0]}"] == flag
        ax.scatter(s.loc[m, "timestamp"], s.loc[m, "fused"], s=2, color=color, label=flag)
    for lv in (0.80, 0.98):
        ax.axhline(lv, color=INK, linestyle="--", linewidth=0.8)
    style(ax, "every horizon", "fused score s")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", markerscale=4, ncol=3)
    save(fig, path, f"5.6b - {LABELS[model]}: fused Worker B score, {station.replace(', Kolkata - WBPCB', '')}, "
                    f"calibration May-Aug 2025")


def station_name(station):
    """Drop the common ", Kolkata - WBPCB" ending."""
    return station.replace(", Kolkata - WBPCB", "")


def station_grid(n):
    """A 2 x 4 grid of panels for the 7 stations (the 8th panel holds the legend)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), facecolor=SURFACE)
    return fig, axes.ravel()


def distance_station_chart(dist, order, model, path):
    """5.2, one model: distance to the group shape per station, season and split (train, validation, calibration).

    What: one panel per station; per season one box per split, no outlier dots
          (a season without hours in a split has no box).
    How:  reads worker_b/distances.csv (split column from step 5.2).
    """
    stations = sorted(dist["station"].unique())
    splits = [p for p in PERIOD_NAMES if p in set(dist["split"])]
    width = 0.8 / len(splits)
    fig, axes = station_grid(len(stations))
    for ax, station in zip(axes, stations):
        s = dist[dist["station"] == station]
        for k, split in enumerate(splits):
            data = [s.loc[(s["season"] == season) & (s["split"] == split), "distance"].to_numpy() for season in order]
            pos = np.arange(len(order)) + (k - (len(splits) - 1) / 2) * width
            keep = [i for i, d in enumerate(data) if len(d)]
            if not keep:
                continue
            bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=width * 0.9, showfliers=False,
                            patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(PERIOD_COLORS[split])
                patch.set_alpha(0.7)
        ax.set_xticks(np.arange(len(order)), order, rotation=20, fontsize=8)
        style(ax, station_name(station), "distance")
    axes[-1].axis("off")
    for split in splits:
        axes[-1].bar(0, 0, color=PERIOD_COLORS[split], label=PERIOD_NAMES[split])
    axes[-1].legend(frameon=False, fontsize=11, labelcolor=INK, loc="center")
    save(fig, path, f"5.2 - {LABELS[model]}: embedding distance per station, season and split")


def labels_station_chart(by_year, horizons, model, path):
    """5.6b, one model: Green / Yellow / Red shares per station, horizon and period (calibration, test).

    What: one panel per station; per horizon one stacked bar per period.
    How:  by_year = {period: label table ranked against all of calibration}.
    """
    stations = sorted(next(iter(by_year.values()))["station"].unique())
    years = list(by_year)
    fig, axes = station_grid(len(stations))
    hx = np.arange(len(horizons))
    for ax, station in zip(axes, stations):
        for k, year in enumerate(years):
            t = by_year[year][by_year[year]["station"] == station]
            pos, bottom = hx + (k - 0.5) * 0.4, np.zeros(len(horizons))
            for flag, color in FLAG_COLORS.items():
                share = np.array([(t[f"label_h{h}"] == flag).mean() * 100 for h in horizons])
                ax.bar(pos, share, 0.36, bottom=bottom, color=color, edgecolor=PERIOD_COLORS[year], linewidth=1.2)
                bottom += share
            for p, h in zip(pos, horizons):
                ax.text(p, 101, f"R{(t[f'label_h{h}'] == 'Red').mean() * 100:.0f}", ha="center", fontsize=6, color=INK)
        ax.set_xticks(hx, [f"+{h}h\n{' | '.join(y[:3] for y in years)}" for h in horizons], fontsize=8)
        ax.set_ylim(0, 108)
        style(ax, station_name(station), "% of samples")
    axes[-1].axis("off")
    for flag, color in FLAG_COLORS.items():
        axes[-1].bar(0, 0, color=color, label=flag)
    for year in years:
        axes[-1].bar(0, 0, color=SURFACE, edgecolor=PERIOD_COLORS[year], linewidth=2,
                     label=f"{PERIOD_NAMES[year]}, left | right bar")
    axes[-1].legend(frameon=False, fontsize=11, labelcolor=INK, loc="center")
    save(fig, path, f"5.6b - {LABELS[model]}: label shares per station, calibration vs test (R = % Red)")


def scores_year_chart(labels, year, model, path):
    """5.6b, one model, one period (calibration or test): daily mean s_embed and s_raw per station, with the daily Red share (+24 h).

    What: one row per station; model-coloured line = s_embed (this model's
          embedding), grey line = s_raw (same for every model), red fill = share
          of that day's hours labelled Red at +24 h.
    How:  daily means of the label table (ranked against all of calibration).
    """
    stations = sorted(labels["station"].unique())
    fig, axes = plt.subplots(len(stations), 1, figsize=(18, 2.1 * len(stations)), facecolor=SURFACE, sharex=True)
    for ax, station in zip(axes, stations):
        s = labels[labels["station"] == station].set_index("timestamp")
        day = s[["s_embed", "s_raw"]].resample("D").mean()
        red = (s["label_h24"] == "Red").resample("D").mean()
        ax.fill_between(red.index, 0, red, color=FLAG_COLORS["Red"], alpha=0.35, linewidth=0, label="Red share (+24 h)")
        ax.plot(day.index, day["s_embed"], color=COLORS[model], linewidth=1.2, label="s_embed (daily mean)")
        ax.plot(day.index, day["s_raw"], color=MUTED, linewidth=1, label="s_raw (daily mean)")
        ax.set_ylim(0, 1)
        style(ax, station_name(station), "score")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=3)
    save(fig, path, f"5.6b - {LABELS[model]}: Worker B scores per station, {PERIOD_NAMES[year]}")


def compare_charts(cfg, names, order, horizons, plots):
    """Comparison files with every model side by side (one chart per Part 5 step + a summary).

    What: 5.2 distance percentiles, 5.3 share p < 0.02, 5.6 correlation and
          share s_raw larger, 5.6b Red share; summary = the key numbers in one figure.
    How:  reads every model's worker_b CSVs.
    """
    fig_dir, x = cfg["figures_dir"], np.arange(len(order))
    width = 0.8 / len(names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), facecolor=SURFACE)
    for k, model in enumerate(names):
        dist = pd.read_csv(worker_dir(cfg, model) / "distances.csv")
        for ax, split in zip(axes, ["train", "calibration"]):
            p99 = dist[dist["split"] == split].groupby("season")["distance"].quantile(0.99).reindex(order)
            ax.bar(x + (k - (len(names) - 1) / 2) * width, p99, width * 0.92, color=COLORS[model], label=LABELS[model])
    for ax, split in zip(axes, ["train", "calibration"]):
        ax.set_xticks(x, order)
        style(ax, f"99th percentile of the distance, {split}", "distance")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, fig_dir / plots["distance"], "5.2 - all models: how far the unusual 1% sit from the normal shape")

    rows = []
    for model in names:
        c = pd.read_csv(worker_dir(cfg, model) / "raw_check.csv")
        rows.append({"model": model, "r": np.corrcoef(c["s_raw"], c["s_embed"])[0, 1],
                     "s_raw larger": (c["s_raw"] > c["s_embed"]).mean() * 100,
                     "fused > 0.98": (c["fused"] > 0.98).mean() * 100})
    raw = pd.DataFrame(rows).set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=SURFACE)
    for ax, (col, title, unit) in zip(axes, [("r", "correlation s_raw vs s_embed", "r"),
                                             ("s_raw larger", "share where s_raw is larger", "%"),
                                             ("fused > 0.98", "share of fused score > 0.98", "%")]):
        bars = ax.bar([LABELS[m] for m in names], raw[col], color=[COLORS[m] for m in names])
        ax.bar_label(bars, fmt="%.2f" if col == "r" else "%.1f", padding=2, fontsize=8, color=INK)
        ax.tick_params(axis="x", labelrotation=20)
        style(ax, title, unit)
    save(fig, fig_dir / plots["raw"], "5.6 - all models: raw safety net vs embedding score (check weeks)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), facecolor=SURFACE)
    hx = np.arange(len(horizons))
    for k, model in enumerate(names):
        s = pd.read_csv(worker_dir(cfg, model) / "label_check.csv")
        s = s[s["season"] == "all"].set_index("horizon").loc[horizons]
        for ax, flag in zip(axes, FLAG_COLORS):
            ax.bar(hx + (k - (len(names) - 1) / 2) * width, s[flag] * 100, width * 0.92, color=COLORS[model],
                   label=LABELS[model])
    g, y = cfg["worker_b"]["green_max"], cfg["worker_b"]["yellow_max"]
    bands = [("Green", f"s ≤ {g:.2f}", g), ("Yellow", f"{g:.2f} < s ≤ {y:.2f}", y - g), ("Red", f"s > {y:.2f}", 1 - y)]
    for ax, (flag, rule, width_s) in zip(axes, bands):
        ax.set_xticks(hx, [f"+{h}h" for h in horizons])
        style(ax, f"{flag}: {rule}", "% of calibration week samples")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="upper right", ncol=5)
    save(fig, fig_dir / plots["labels"], "5.6b - all models: share of calibration week hours in each band of the "
                                         "fused score s (calibration May-Aug 2025)")


def main():
    """Draw every Part 5 chart: per model in figures/<model>/, comparisons in figures/.

    What: per model: 5.2 distance, 5.3 p-value check, 5.6 raw vs embed, 5.6b label
          shares and timeline; comparison: one chart per step.
    How:  reads the CSVs written by the Part 5 steps; file names from config.yaml
          worker_b.plots. No test data.
    """
    cfg = load_config()
    w, horizons, order = cfg["worker_b"], cfg["horizons"], season_order(cfg)
    plots, names, fig_dir = w["plots"], cfg["forecasters"]["names"], cfg["figures_dir"]
    for model in names:
        wd, md = worker_dir(cfg, model), fig_dir / model
        distance_chart(pd.read_csv(wd / "distances.csv"), order, model, md / plots["distance"])
        raw_chart(pd.read_csv(wd / "raw_check.csv"), model, md / plots["raw"])
        labels_chart(pd.read_csv(wd / "label_check.csv"), order, horizons, model, md / plots["labels"], flag_rules(w))
        calib_labels = pd.read_csv(wd / w["labels_file"], parse_dates=["timestamp"])
        timeline_chart(calib_labels, w["timeline_station"], horizons, model, md / plots["timeline"])
        distance_station_chart(pd.read_csv(wd / "distances.csv"), order, model, md / plots["distance_station"])
        by_year = {"calibration": calib_labels}
        if (wd / "worker_b_test.csv").exists():              # written by step 7.1 (test Sep-Dec 2025)
            by_year["test"] = pd.read_csv(wd / "worker_b_test.csv", parse_dates=["timestamp"])
        labels_station_chart(by_year, horizons, model, md / plots["labels_station"])
        for year, table in by_year.items():
            scores_year_chart(table, year, model, md / plots["scores_year"].format(year=year))
    compare_charts(cfg, names, order, horizons, plots)
    print(f"saved: per model in {fig_dir}/<model>/, comparisons in {fig_dir}")


if __name__ == "__main__":
    main()
