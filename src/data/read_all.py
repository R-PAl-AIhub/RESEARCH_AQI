"""Read every CPCB file, join them, and put each station on a complete hourly grid."""
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.read_cpcb import read_one


def find_files(raw_dir):
    """List every Excel file under raw_dir.

    What: returns a sorted list of .xlsx paths, searching all sub-folders.
    How:  Path.rglob("*.xlsx") walks every folder below raw_dir. Files starting
          with "~$" are skipped: Excel creates these temporary lock files while a
          workbook is open, and they are not real data.
    """
    return sorted(p for p in Path(raw_dir).rglob("*.xlsx") if not p.name.startswith("~$"))


def to_hourly_grid(g):
    """Give one station a row for every hour from Jan 1 of its first year to Dec 31 of its last.

    What: returns the station's table with no missing hours; hours that were not
          in any file become rows of NaN, and row_present says which is which.
    How:  1. build the full list of hours with pd.date_range(freq="h");
          2. row_present = True for hours that appear in the station's data;
          3. reindex on that full list, which inserts empty rows for absent hours;
          4. refill the station name on the inserted rows.
    """
    station = g["station"].iloc[0]
    years = g["time"].dt.year
    full = pd.date_range(f"{years.min()}-01-01 00:00", f"{years.max()}-12-31 23:00",
                         freq="h", name="time")

    g = g.set_index("time")
    present = full.isin(g.index)
    g = g.reindex(full)
    g["station"] = station
    g.insert(0, "row_present", present)
    return g.reset_index()


def read_all(raw_dir):
    """Build one hourly table for all stations.

    What: returns (hourly, report). hourly has columns station, time,
          row_present, then the variables; report holds the counts for the check.
    How:  1. read every file with read_one and stack them (pd.concat lines up
             columns by name, so files with different column orders still match);
          2. drop exact duplicate rows (every column identical);
          3. if the same station+hour still appears twice with DIFFERENT values,
             stop with an error instead of guessing which one is right;
          4. put each station on its full hourly grid with to_hourly_grid.
    """
    files = find_files(raw_dir)
    tables = [read_one(f)[0] for f in files]
    raw = pd.concat(tables, ignore_index=True)

    dedup = raw.drop_duplicates()
    clashes = int(dedup.duplicated(["station", "time"]).sum())
    if clashes:
        raise ValueError(f"{clashes} station+hour pairs appear twice with different values")

    hourly = pd.concat(
        [to_hourly_grid(g) for _, g in dedup.groupby("station", sort=True)],
        ignore_index=True,
    )
    columns = ["station", "time", "row_present"] + [c for c in raw.columns if c not in ("station", "time")]
    hourly = hourly[columns]

    report = {
        "files": files,
        "raw_rows": len(raw),
        "exact_duplicates": len(raw) - len(dedup),
        "station_hour_clashes": clashes,
    }
    return hourly, report


def main():
    """Run step 2.2: build the hourly table, save it, and print the check.

    What: saves processed_dir/cpcb_hourly_file and prints files found, raw rows,
          gridrows, duplicate count and the date range per station.
    How:  reads raw_dir, processed_dir and the file name from config.yaml (so no
          path is written in code), calls read_all, writes CSV, then prints a
          per-station summary from the saved table.
    """
    cfg = load_config()
    hourly, report = read_all(cfg["raw_dir"])

    out = cfg["processed_dir"] / cfg["cpcb_hourly_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out, index=False)

    print(f"files found:            {len(report['files'])}")
    print(f"raw rows:               {report['raw_rows']}")
    print(f"exact duplicates:       {report['exact_duplicates']}")
    print(f"station+hour clashes:   {report['station_hour_clashes']}")
    print(f"grid rows (all):        {len(hourly)}")
    print(f"duplicate station+time in grid: {int(hourly.duplicated(['station', 'time']).sum())}")
    print(f"saved to:               {out}")

    summary = hourly.groupby("station").agg(
        first=("time", "min"),
        last=("time", "max"),
        grid_rows=("time", "size"),
        present=("row_present", "sum"),
    )
    summary["absent"] = summary["grid_rows"] - summary["present"]
    print("\nper station:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
