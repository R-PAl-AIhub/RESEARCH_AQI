"""Teacher update (2026-11-02): coverage tables per station, year, model and imputation run (report only).

A. Published range: the 90% range of step 7.1 (CQR + adaptive, the base range, stored for every test hour,
   also Red ones) - it exists only on test (Sep-Dec 2025). Same definition as method (c) of step 7.2.
B. The model's own 90% band (quantile low-high, before any conformal correction) - every split, so every year.
Coverage = share of real values inside [lower, upper]. Nothing here chooses a setting.
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
from src.temp_out import save_text

SECONDARY = "#52514e"
GAP = LinearSegmentedColormap.from_list("gap", ["#e34948", "#f0efec", "#2a78d6"])   # too narrow - 90% - too wide
ALL = "all stations"


def inside_rows(table, lower, upper, keys, cfg):
    """Hits per row for every target x horizon: long table with the key columns, target, horizon, inside (0/1)."""
    parts = []
    for target in cfg["targets"]:
        for h in cfg["horizons"]:
            k = f"{short(target)}_h{h}"
            y, lo, hi = table[f"target_{k}"], table[f"{lower}{k}"], table[f"{upper}{k}"]
            ok = y.notna() & lo.notna() & hi.notna()
            parts.append(table.loc[ok, keys].assign(target=target, horizon=h,
                                                    inside=((y >= lo) & (y <= hi))[ok].astype(float)))
    return pd.concat(parts, ignore_index=True)


def coverage_of(hits, keys):
    """Coverage and n per group, plus the pooled 'all stations' group (pooled over hours, not a mean of stations)."""
    by = hits.groupby(keys + ["station", "target", "horizon"])["inside"].agg(coverage="mean", n="size").reset_index()
    pooled = hits.groupby(keys + ["target", "horizon"])["inside"].agg(coverage="mean", n="size").reset_index()
    return pd.concat([by, pooled.assign(station=ALL)], ignore_index=True)


def published_table(cfg):
    """A: coverage of the step 7.1 90% range on test, per run, model, station, target and horizon."""
    out = []
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        for model in cfg["forecasters"]["names"]:
            dec = pd.read_csv(rc["forecasts_dir"] / model / cfg["evaluation"]["flags_file"])
            truth = pd.read_csv(rc["forecasts_dir"] / model / "quantile" / "test.csv",
                                usecols=lambda c: c in ("station", "timestamp") or c.startswith("target_"))
            table = dec.merge(truth, on=["station", "timestamp"], how="inner")
            hits = inside_rows(table, "base_lower_", "base_upper_", ["station"], cfg)
            out.append(coverage_of(hits.assign(method=method, model=model), ["method", "model"]))
    return pd.concat(out, ignore_index=True)


def band_table(cfg):
    """B: coverage of the model's own 90% band, per run, model, station, year, target and horizon (every split)."""
    out = []
    for method in cfg["imputation"]["methods"]:
        rc = run_config(method)
        for model in cfg["forecasters"]["names"]:
            hits = []
            for split in cfg["metrics"]["splits"]:
                t = pd.read_csv(rc["forecasts_dir"] / model / "quantile" / f"{split}.csv", parse_dates=["timestamp"])
                t["year"] = t["timestamp"].dt.year
                hits.append(inside_rows(t, "low_", "high_", ["station", "year"], cfg))
            out.append(coverage_of(pd.concat(hits).assign(method=method, model=model), ["method", "model", "year"]))
    return pd.concat(out, ignore_index=True)


def summary(table, keys):
    """Mean over the horizons per group (n = hours of the first horizon)."""
    out = table.groupby(keys, sort=False)["coverage"].mean().to_frame()
    out["n"] = table[table["horizon"] == table["horizon"].min()].set_index(keys)["n"]
    return out.reset_index()


def heat_table(cells, columns, col_labels, groups, cfg, title, notes, path):
    """Coverage as a shaded table: rows = imputation run x model, columns = the given (group, station) pairs.

    What: the value (%) is written in every cell; colour = distance from the 90% aim (red = too narrow,
          grey = on target, blue = too wide, clipped at ±shade_points); gaps between runs and column groups.
    How:  cells = DataFrame indexed by (method, model) with one column per entry of `columns`.
    """
    methods, models = cfg["imputation"]["methods"], cfg["forecasters"]["names"]
    rows = [(m, n) for m in methods for n in models]
    grid = cells.reindex(index=pd.MultiIndex.from_tuples(rows), columns=columns).to_numpy() * 100
    aim, span = (1 - cfg["alpha"]) * 100, cfg["coverage"]["shade_points"]
    fig, ax = plt.subplots(figsize=(3.6 + 0.6 * len(columns), 0.3 * len(rows) + 3.6), facecolor=SURFACE)
    ax.imshow(grid - aim, cmap=GAP, vmin=-span, vmax=span, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(columns)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v - aim) > 0.7 * span else INK)
    ax.set_yticks(range(len(rows)), [LABELS[n] for _, n in rows], fontsize=8)
    for k, m in enumerate(methods):
        ax.annotate(m, xy=(0, k * len(models) + (len(models) - 1) / 2), xycoords=("axes fraction", "data"),
                    xytext=(-78, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK,
                    fontweight="bold")
        if k:
            ax.axhline(k * len(models) - 0.5, color=SURFACE, linewidth=3)
    ax.set_xticks(range(len(columns)), col_labels, rotation=90, fontsize=8)
    ax.xaxis.tick_top()
    starts = [j for j in range(len(columns)) if j == 0 or columns[j][0] != columns[j - 1][0]]
    for j in starts[1:]:
        ax.axvline(j - 0.5, color=SURFACE, linewidth=4)
    if groups:
        for j0, j1 in zip(starts, starts[1:] + [len(columns)]):
            ax.annotate(groups[columns[j0][0]], xy=((j0 + j1 - 1) / 2, 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 150), textcoords="offset points", ha="center", va="bottom", fontsize=11,
                        color=INK, fontweight="bold")
    ax.tick_params(length=0, labelcolor=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    for k, line in enumerate(notes):
        fig.text(0.01, 0.03 - 0.018 * k, line, fontsize=8.5, color=SECONDARY if k == 0 else MUTED)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.72 if groups else 0.8, bottom=0.06)
    save_figure(fig, path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Build both coverage tables (CSV per horizon + summary) and one picture per target for each.

    What: prints a check against step 7.2 (GRU, ffill, first target, +1 h, all test hours) and the
          all-stations coverage per run and model for both ranges.
    How:  published_table / band_table -> summary -> heat_table.
    """
    cfg = load_config()
    c = cfg["coverage"]
    pub = published_table(cfg)
    band = band_table(cfg)
    stations = sorted(s for s in pub["station"].unique() if s != ALL)
    names = {s: s.split(",")[0] for s in stations} | {ALL: "All stations"}
    pub_sum = summary(pub, ["method", "model", "target", "station"])
    band_sum = summary(band, ["method", "model", "target", "year", "station"])
    for df, key in ((pub, "published_file"), (pub_sum, "published_summary_file"), (band, "band_file"),
                    (band_sum, "band_summary_file")):
        df.round(4).to_csv(cfg["figures_dir"] / c[key], index=False)
    years = sorted(band_sum["year"].unique())
    for target in cfg["targets"]:
        p = pub_sum[pub_sum["target"] == target].pivot_table(index=["method", "model"], columns="station",
                                                           values="coverage")
        p.columns = [("test", s) for s in p.columns]
        cols = [("test", s) for s in stations + [ALL]]
        heat_table(p, cols, [names[s] for _, s in cols], None, cfg,
                   f"Coverage of the published 90% range at every station: {target} (test Sep–Dec 2025)",
                   ["Cell = % of real values inside the step 7.1 range (CQR + adaptive), every test hour, mean over "
                    "the 5 horizons. Aim 90%. Red = range too narrow there, blue = wider than needed.",
                    "The range exists only on test (it is built from calibration). Test is reported only; nothing "
                    "was chosen from it. Per-horizon values and hour counts: coverage published CSV."],
                   cfg["figures_dir"] / c["published_plot"].format(target=short(target)))
        b = band_sum[band_sum["target"] == target].pivot_table(index=["method", "model"], columns=["year", "station"],
                                                             values="coverage")
        cols = [(y, s) for y in years for s in stations + [ALL]]
        heat_table(b, cols, [names[s] for _, s in cols], {y: str(y) for y in years}, cfg,
                   f"Coverage of the model's own 90% band, every station and year: {target}",
                   ["Cell = % of real values inside the forecaster's own low–high band (before the conformal "
                    "correction), mean over the 5 horizons. Aim 90%. Red = band too narrow, blue = too wide.",
                    "2023–24 = training years (the band was fitted there); 2025 = validation, calibration and test "
                    "months together. Per-horizon values and hour counts: coverage band CSV."],
                   cfg["figures_dir"] / c["band_plot"].format(target=short(target)))
    first, h1 = cfg["targets"][0], cfg["horizons"][0]
    chk = pub[(pub["method"] == "ffill") & (pub["model"] == "gru") & (pub["station"] == ALL)
              & (pub["target"] == first) & (pub["horizon"] == h1)].iloc[0]
    t1 = pd.read_csv(run_config("ffill")["figures_dir"] / "gru" / cfg["evaluation"]["table1_file"])
    t1 = t1[(t1["target"] == first) & (t1["horizon"] == h1) & (t1["method"] == "(c) CQR + adaptive")]
    pooled_72 = (t1["coverage"] * t1["samples"]).sum() / t1["samples"].sum()
    every = lambda s, cols: s[s["station"] == ALL].pivot_table(index=["method", "model"], columns=cols,
                                                              values="coverage").round(3).to_string()
    text = "\n".join([f"check (GRU, ffill, {first} +{h1} h, all test hours): this table {chk['coverage']:.4f} "
                      f"(n {int(chk['n'])}), step 7.2 normal + smoggiest pooled {pooled_72:.4f}", "",
                      "published 90% range, test, all stations (mean over horizons):", every(pub_sum, "target"), "",
                      "model's own band, all stations, per year (mean over horizons):",
                      every(band_sum, ["target", "year"])])
    print(text)
    save_text(cfg, "coverage_table_report.txt", text)


if __name__ == "__main__":
    main()
