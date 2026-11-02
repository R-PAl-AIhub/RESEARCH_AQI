"""Step 2.4 (PROJECT_2): find stuck-sensor PM readings and chart them. No value is changed."""
import matplotlib

matplotlib.use("Agg")                     # draw to a file, no window needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plot_style import save_figure
from src.temp_out import save_text

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
COLOURS = {"flat PM2.5": "#2a78d6", "flat PM10": "#eb6834"}


def check_hourly_order(df):
    """Make sure each station's rows are consecutive hours, in time order.

    What: raises an error if any station has a jump other than exactly 1 hour.
    How:  within each station, the difference between one row's time and the
          previous row's time must be 1 hour. The stuck rule relies on
          "the row above = the hour before", so this must hold.
    """
    steps = df.groupby("station")["time"].diff().dropna()
    if not (steps == pd.Timedelta("1h")).all():
        raise ValueError("rows are not consecutive hours within each station")


def flat_flags(df, variables, min_run):
    """Stuck-sensor rule: flag_flat_<var> (PM2.5 and PM10 only). Copied from PROJECT.

    What: True on every hour of a run where the same value repeats min_run or
          more hours in a row. Also returns a table with one row per run.
    How:  1. compare each value with the hour before at the same station;
          2. a "new run" starts whenever they differ, or either one is missing
             (missing never equals anything, so a gap always breaks a run);
          3. counting new-run starts gives every row a run number;
          4. the size of its run number = run length; flag if >= min_run.
    """
    flags, runs = {}, []
    for var in variables:
        value = df[var]
        previous = value.groupby(df["station"]).shift(1)
        new_run = ~(value == previous)
        run_id = new_run.cumsum()
        run_length = run_id.map(run_id.value_counts())
        flag = (run_length >= min_run) & value.notna()
        flags[f"flag_flat_{var}"] = flag

        hits = df.loc[flag, ["station", "time", var]].assign(run_id=run_id[flag])
        table = hits.groupby("run_id").agg(
            station=("station", "first"),
            start=("time", "min"),
            end=("time", "max"),
            hours=("time", "size"),
            value=(var, "first"),
        )
        table.insert(0, "variable", var)
        runs.append(table)
    return pd.DataFrame(flags), pd.concat(runs, ignore_index=True)


def flag_counts(df, flags):
    """Stuck hours per station, year and variable.

    What: one row per station+year, one column per flag ("flat PM2.5", "flat PM10").
    How:  True counts as 1, so summing a flag column per station+year gives the
          number of stuck hours there.
    """
    keys = [df["station"].str.replace(", Kolkata - WBPCB", ""), df["time"].dt.year.rename("year")]
    counts = flags.groupby(keys).sum()
    counts.columns = [c.removeprefix("flag_").replace("_", " ", 1) for c in counts.columns]
    return counts


def plot_counts(counts, out):
    """Chart: stuck-sensor PM hours per station and year.

    What: one group per station+year with one bar for PM2.5 and one for PM10,
          and the hour count above every non-zero bar. Values are only counted
          here, never changed.
    How:  the bars of each flag sit side by side around each station+year tick.
    """
    labels = [f"{s} {y}" for s, y in counts.index]
    x = np.arange(len(counts))
    width = 0.8 / len(counts.columns)
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=SURFACE)
    for i, col in enumerate(counts.columns):
        pos = x + (i - (len(counts.columns) - 1) / 2) * width
        bars = ax.bar(pos, counts[col], width, label=col, color=COLOURS.get(col, MUTED))
        ax.bar_label(bars, labels=[f"{v:,}" if v else "" for v in counts[col]], fontsize=7, color=INK, padding=1)

    ax.set_title("Stuck-sensor PM hours (same value 12+ hours) per station and year (values kept)",
                 color=INK, fontsize=11, loc="left")
    ax.set_ylabel("hours", color=MUTED)
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def main():
    """Run step 2.4: find stuck PM readings, save the run list, counts and chart.

    What: applies the stuck-sensor rule (settings in config flags.flat) to the
          step 2.2 hourly table and saves figures/flat_runs.csv, flag_counts.csv
          and flag_check.png. No value is changed and no data table is written:
          the readings are kept as they are.
    How:  file names and limits come from config.yaml; the hourly table is only read.
    """
    cfg = load_config()
    rule = cfg["flags"]["flat"]
    hourly = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_hourly_file"], parse_dates=["time"])
    check_hourly_order(hourly)

    flat, runs = flat_flags(hourly, rule["variables"], rule["min_run_hours"])
    counts = flag_counts(hourly, flat)

    fig_dir = cfg["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(fig_dir / cfg["flat_runs_file"], index=False)
    counts.to_csv(fig_dir / cfg["flag_counts_file"])
    plot_counts(counts, fig_dir / cfg["flag_plot_file"])

    pd.set_option("display.width", 200)
    lines = [
        f"rows checked: {len(hourly)}  (values kept, nothing changed)",
        f"\nstuck rule: same PM value {rule['min_run_hours']}+ hours in a row",
        "\nstuck hours, all stations and years:",
        counts.sum().to_string(),
        f"stuck runs: {len(runs)}",
        "\nstuck hours per station and year:",
        counts.to_string(),
        f"\nsaved to: {fig_dir}  ({cfg['flat_runs_file']}, {cfg['flag_counts_file']}, {cfg['flag_plot_file']})",
    ]
    text = "\n".join(lines)
    print(text)

    save_text(cfg, "2.4_flags_report.txt", text)
    print(f"temp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
