"""Step 2.5 (PROJECT_2): count out-of-range readings, add 0/1 stuck-PM columns, no value changed; save the table."""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean.flags import flat_flags
from src.config import load_config
from src.plot_style import save_figure
from src.temp_out import save_table, save_text

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
COLOURS = ["#2a78d6", "#eb6834", "#3a9e5c", "#8e5bd0", "#d4a017", "#cc3d6b", "#3fa7b5",
           "#7a6a58", "#5b8c2a", "#b0413e", "#4a5fa8", "#c27c0e", "#6b6b6b", "#9c4f96"]


def out_of_range(df, limits):
    """True where a reading is outside its [low, high] range.

    What: returns a true/false table, one column per variable in config flags.range.
    How:  compares each column with its limits; a null limit means "no limit on
          that side". A missing value compares as False, so it is never counted.
          Names in the config that are not columns stop the run with a clear message.
    """
    unknown = [var for var in limits if var not in df.columns]
    if unknown:
        raise ValueError(f"config flags.range names not in the data: {unknown}")
    bad = {}
    for var, (low, high) in limits.items():
        mask = pd.Series(False, index=df.index)
        if low is not None:
            mask |= df[var] < low
        if high is not None:
            mask |= df[var] > high
        bad[var] = mask
    return pd.DataFrame(bad)


def range_counts(df, bad):
    """In-range and out-of-range readings per station, year and variable.

    What: one row per station+year+variable with columns in_range, out_of_range
          and pct_in_range (% of the readings that had a value).
    How:  a reading "has a value" if it is not missing; out_of_range comes from
          the true/false table; in_range = has a value and not out of range.
    """
    keys = [df["station"].str.replace(", Kolkata - WBPCB", ""), df["time"].dt.year.rename("year")]
    present = df[bad.columns].notna()
    out = bad.groupby(keys).sum().stack().rename("out_of_range")
    total = present.groupby(keys).sum().stack().rename("with_value")
    table = pd.concat([total - out, out], axis=1, keys=["in_range", "out_of_range"])
    table.index.names = ["station", "year", "variable"]
    table["pct_in_range"] = (table["in_range"] / (table["in_range"] + table["out_of_range"]) * 100).round(3)
    return table


def plot_counts(table, out):
    """Chart: readings outside the official range, per station and year.

    What: one bar per station+year, stacked by variable (only variables with at
          least one out-of-range reading get a colour); above each bar the total
          out-of-range count; the x label also gives the % of readings in range.
    How:  sums the table per station+year; bars are drawn variable by variable,
          each starting where the previous one ended (the "bottom" argument).
    """
    per_sy = table["out_of_range"].unstack("variable")
    per_sy = per_sy.loc[:, per_sy.sum() > 0]
    totals = table.groupby(["station", "year"])[["in_range", "out_of_range"]].sum()
    pct = totals["in_range"] / totals.sum(axis=1) * 100

    labels = [f"{s} {y} ({p:.2f}% ok)" for (s, y), p in zip(per_sy.index, pct)]
    x = np.arange(len(per_sy))
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=SURFACE)
    bottom = np.zeros(len(per_sy))
    for i, var in enumerate(per_sy.columns):
        ax.bar(x, per_sy[var], bottom=bottom, label=var, color=COLOURS[i % len(COLOURS)])
        bottom += per_sy[var].to_numpy()
    for xi, b in zip(x, bottom):
        ax.text(xi, b, f"{int(b):,}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_title("Readings outside the official range (values kept), per station and year", color=INK, fontsize=11, loc="left")
    ax.set_ylabel("readings outside range", color=MUTED)
    ax.set_ylim(0, max(bottom.max() * 1.25, 1))
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, title="variable", fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def main():
    """Run step 2.5: count out-of-range readings, mark stuck PM hours 0/1, save, chart and print the check.

    What: reads the step 2.2 hourly table, counts readings outside their official
          range WITHOUT changing any value, adds stuck_pm25 and stuck_pm10 (1 = that
          hour is inside a stuck run, config flags.flat: same value 12+ hours, else
          0), drops the columns in config clean.drop_columns (as PROJECT step 2.5),
          saves processed_dir/cpcb_clean.csv (CSV only), the count table and the
          chart, and copies the printed check into temp/.
    How:  the range check runs on a separate copy converted to the CPCB unit
          (config flags.unit_factors, e.g. BP hPa -> mmHg); the stuck runs come from
          flat_flags (step 2.4 rule). Every row stays and every source value is kept.
          The input file is only read, never changed.
    """
    cfg = load_config()
    hourly = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_hourly_file"], parse_dates=["time"])
    check = hourly.copy()                           # only for the range check; never saved
    for var, factor in cfg["flags"]["unit_factors"].items():
        check[var] = check[var] * factor            # to the CPCB unit (e.g. BP hPa -> mmHg)

    bad = out_of_range(check, cfg["flags"]["range"])
    rule = cfg["flags"]["flat"]
    stuck, _ = flat_flags(hourly, rule["variables"], rule["min_run_hours"])
    marks = {f"stuck_{var.lower().replace('.', '')}": stuck[f"flag_flat_{var}"].astype(int)
             for var in rule["variables"]}           # stuck_pm25, stuck_pm10

    clean = hourly.drop(columns=cfg["clean"]["drop_columns"])   # drop columns as PROJECT step 2.5
    values_kept = clean.equals(hourly[clean.columns])
    clean = clean.assign(**marks)
    marks_per_station = clean.groupby(clean["station"].str.replace(", Kolkata - WBPCB", ""))[list(marks)].sum()

    out = cfg["processed_dir"] / cfg["cpcb_clean_file"]
    clean.to_csv(out, index=False)

    table = range_counts(hourly, bad)
    counts_out = cfg["figures_dir"] / cfg["range_counts_file"]
    table.to_csv(counts_out)
    plot_out = cfg["figures_dir"] / cfg["range_plot_file"]
    plot_counts(table, plot_out)

    per_var = table.groupby("variable")[["in_range", "out_of_range"]].sum()
    per_var["pct_in_range"] = (per_var["in_range"] / per_var.sum(axis=1) * 100).round(3)

    pd.set_option("display.width", 200)
    lines = [
        f"rows: {len(clean)}",
        "\nreadings per variable (all stations, all years):",
        per_var.to_string(),
        f"\nreadings outside range: {int(bad.values.sum())}  (counted only, values kept)",
        f"unit conversions used for the check only (not saved): {cfg['flags']['unit_factors']}",
        f"\nstuck marks added (1 = same value {rule['min_run_hours']}+ h): "
        + ", ".join(f"{k} = 1 on {int(v.sum())} h" for k, v in marks.items()),
        "hours marked 1 per station:",
        marks_per_station.to_string(),
        f"every source value kept (all {len(clean)} rows): {values_kept}",
        f"dropped columns (config clean.drop_columns): {cfg['clean']['drop_columns']}",
        f"kept columns: {list(clean.columns)}",
        f"\nsaved to: {out}",
        f"counts:   {counts_out}",
        f"plot:     {plot_out}",
    ]
    text = "\n".join(lines)
    print(text)

    save_text(cfg, "2.5_range_clean_report.txt", text)
    save_table(cfg, "2.5_range_per_variable.csv", per_var)
    print(f"temp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
