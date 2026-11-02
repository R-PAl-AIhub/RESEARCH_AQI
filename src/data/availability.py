"""Step 2.3: how much data does each station actually have, per year and variable?"""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure, table_figure
from src.temp_out import save_table, save_text

ID_COLUMNS = ("station", "time", "row_present")
YEAR_COLOURS = ["#2a78d6", "#eb6834", "#3a9e5c"]   # one per year
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"


def availability_table(hourly):
    """Percentage of hours with a real (non-missing) value, per station, year and variable.

    What: returns a table with one row per station+year and one column per
          variable, plus "row_present" (% of hours that existed in the file at all).
    How:  .notna() turns every cell into True (has a value) or False (missing).
          The mean of True/False per group is the share of hours with a value;
          times 100 gives a percentage. The denominator is every hour of the year
          on the grid (8760, or 8784 in a leap year), so absent hours count as missing.
    """
    variables = [c for c in hourly.columns if c not in ID_COLUMNS]
    flags = hourly[variables].notna()
    flags["row_present"] = hourly["row_present"]
    flags["station"] = hourly["station"]
    flags["year"] = hourly["time"].dt.year
    table = flags.groupby(["station", "year"]).mean() * 100
    return table[["row_present"] + variables].round(1)


def wind_speed_columns(hourly):
    """Find any wind-speed column.

    What: returns the list of column names that look like wind speed.
    How:  checks every column name, ignoring case, for "WS" exactly or the words
          "wind speed". An empty list means no wind speed was delivered.
    """
    return [c for c in hourly.columns
            if c.strip().upper() == "WS" or "WIND SPEED" in c.upper()]


def first_pm25_month(hourly, min_pct):
    """First month in which each station has real PM2.5 data.

    What: returns one row per station with two months:
          first_any  = first month with at least one PM2.5 value;
          first_real = first month where at least min_pct % of hours have PM2.5
                       (so one stray reading does not count as "real data").
    How:  groups hours by station and calendar month, computes the % of hours with
          PM2.5, then picks the earliest month passing each rule.
    """
    df = pd.DataFrame({
        "station": hourly["station"],
        "month": hourly["time"].dt.to_period("M"),
        "has_pm25": hourly["PM2.5"].notna(),
    })
    pct = df.groupby(["station", "month"])["has_pm25"].mean() * 100

    rows = []
    for station, s in pct.groupby(level="station"):
        s = s.droplevel("station")
        any_months = s.index[s > 0]
        real_months = s.index[s >= min_pct]
        rows.append({
            "station": station,
            "first_any": str(any_months[0]) if len(any_months) else "none",
            "first_real": str(real_months[0]) if len(real_months) else "none",
        })
    return pd.DataFrame(rows).set_index("station")


def target_hours(hourly, target):
    """Hours with a value for one target, per station and year.

    What: returns a table with one row per station and one column per year.
    How:  .notna() marks hours with a value; summing per station+year counts
          them; unstack puts years side by side; the long ", Kolkata - WBPCB"
          ending is cut from station names so chart labels stay short.
    """
    counts = (hourly[target].notna()
              .groupby([hourly["station"], hourly["time"].dt.year.rename("year")]).sum()
              .unstack("year"))
    counts.index = counts.index.str.replace(", Kolkata - WBPCB", "")
    return counts


def plot_target_hours(hourly, targets, out, ncols=1):
    """Bar charts: hours with a value, per station and year, one panel per variable.

    What: for each variable (the targets, or every kept column) one panel with a
          group of bars per station and one bar per year, the hour count written
          inside each bar, and a dashed line at all hours of a year (8760).
          Panels fill a grid of `ncols` columns. Returns the counts as one table
          (columns = variable, year).
    How:  target_hours gives each panel's numbers; bars are shifted left/right
          of each station's tick so the years sit next to each other.
    """
    nrows = -(-len(targets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 5 * nrows), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    tables = {}
    for ax, target in zip(axes.ravel(), targets):
        counts = target_hours(hourly, target)
        tables[target] = counts
        x = np.arange(len(counts))
        width = 0.8 / len(counts.columns)
        for i, year in enumerate(counts.columns):
            pos = x + (i - (len(counts.columns) - 1) / 2) * width
            bars = ax.bar(pos, counts[year], width, label=str(year), color=YEAR_COLOURS[i % len(YEAR_COLOURS)])
            ax.bar_label(bars, labels=[f"{v:,}" for v in counts[year]], label_type="center",
                         fontsize=8, color="white", rotation=90)
        ax.axhline(8760, color=MUTED, linestyle="--", linewidth=1, label="all hours in a year (8760)")

        ax.set_title(f"Hours with a {target} value, per station and year", color=INK, fontsize=11, loc="left")
        ax.set_ylabel("hours", color=MUTED)
        ax.set_ylim(0, 10500)
        ax.set_xticks(x, counts.index, rotation=15, ha="right")
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, ncol=len(counts.columns) + 1, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)
    return pd.concat(tables, axis=1, names=["target", "year"])


def missing_table(hourly, variables, by_station=False):
    """Missing hours per year and variable, as a count and a percentage (the style of a paper's data table).

    What: returns a table with one row per variable (or per station + variable) and,
          for each year, two columns: "<year> missing" (hours with no value) and
          "<year> %" (that count over all hours of the year).
    How:  the hourly grid of step 2.2 has every hour of every year, so absent hours
          count as missing. City-wide, the base is hours of the year x stations.
    """
    keys = [hourly["station"]] if by_station else []
    flags = hourly[variables].isna()
    year = hourly["time"].dt.year.rename("year")
    missing = flags.groupby(keys + [year]).sum()
    total = flags.groupby(keys + [year]).size()
    pct = missing.div(total, axis=0) * 100
    level = ["station", "year"] if by_station else ["year"]
    long = pd.concat({"missing": missing.stack(), "%": pct.stack().round(1)}, axis=1)
    long.index.names = level + ["variable"]
    table = long.unstack("year")
    table.columns = [f"{y} {kind}" for kind, y in table.columns]
    order = [f"{y} {kind}" for y in sorted(year.unique()) for kind in ("missing", "%")]
    table = table[order]
    if not by_station:
        table = table.reindex(variables)
    return table


def main():
    """Run step 2.3: print the three reports and save them.

    What: prints % present per station/year/variable, the wind-speed answer,
          each station's first PM2.5 month and the hour counts of each target. Saves
          the table as figures/availability.csv (the step's official output), the bar
          charts of target hours, and copies everything into temp/ for inspection.
    How:  reads the hourly table saved in step 2.2 (path and file names come from
          config.yaml), calls the three functions above, collects the printed lines
          in a list so the same text goes to the screen and to temp/.
    """
    cfg = load_config()
    hourly = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_hourly_file"], parse_dates=["time"])

    table = availability_table(hourly)
    first_month = first_pm25_month(hourly, cfg["first_month_min_pct"])
    ws = wind_speed_columns(hourly)

    out = cfg["figures_dir"] / cfg["availability_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    plot = cfg["figures_dir"] / cfg["availability_plot_file"]
    counts = plot_target_hours(hourly, cfg["targets"], plot)
    kept = [c for c in hourly.columns if c not in ID_COLUMNS and c not in cfg["clean"]["drop_columns"]]
    plot_all = cfg["figures_dir"] / cfg["availability_all_plot_file"]
    counts_all = plot_target_hours(hourly, kept, plot_all, ncols=3)          # every kept variable, for viewing
    counts_all.to_csv(cfg["figures_dir"] / cfg["availability_all_file"])
    missing = missing_table(hourly, kept)                                    # paper-style missing table
    missing.to_csv(cfg["figures_dir"] / cfg["missing_table_file"])
    table_figure(missing, "Missing hours per year and variable, all 7 stations together (count and % of all hours)",
                 cfg["figures_dir"] / cfg["missing_table_plot_file"], pct_cols=[c for c in missing.columns if c.endswith("%")])
    missing_table(hourly, kept, by_station=True).to_csv(cfg["figures_dir"] / cfg["missing_station_file"])

    lines = [
        "% of hours with a value (station x year):",
        table.to_string(),
        f"\nwind speed column (WS): {'YES -> ' + ', '.join(ws) if ws else 'NO, not in any file'}",
        f"\nfirst PM2.5 month (real = at least {cfg['first_month_min_pct']}% of hours):",
        first_month.to_string(),
        f"\nhours with a value, targets {', '.join(cfg['targets'])} (station x year):",
        counts.to_string(),
        f"\nsaved to: {out}",
        f"plot:     {plot}",
        f"all kept variables ({len(kept)}): {plot_all}",
        "\nmissing hours per year and variable (all stations together):",
        missing.to_string(),
    ]
    text = "\n".join(lines)
    print(text)

    save_table(cfg, "2.3_availability.csv", table)
    save_table(cfg, "2.3_first_pm25_month.csv", first_month)
    save_table(cfg, "2.3_target_hours.csv", counts)
    report = save_text(cfg, "2.3_availability_report.txt", text)
    print(f"temp copies in: {report.parent}")


if __name__ == "__main__":
    main()
