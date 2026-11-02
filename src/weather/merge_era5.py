"""Step 2.7 (PROJECT_2): give every station-hour its ERA5 wind only (u10, v10, net speed)."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from src.temp_out import save_table, save_text

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)   # India Standard Time = UTC + 5:30
WIND = ["u10", "v10"]                            # the only ERA5 variables read


def load_stations(path, names_in_data):
    """Read data/stations.csv and check it covers exactly the stations in the data.

    What: returns a table with station, latitude, longitude.
    How:  pandas read_csv; the station names must be the same as the "station"
          column of cpcb_clean.csv, otherwise the join would silently lose a
          station, so any difference stops the run with the names listed.
    """
    stations = pd.read_csv(path)
    missing, extra = set(names_in_data) - set(stations["station"]), set(stations["station"]) - set(names_in_data)
    if missing or extra:
        raise SystemExit(f"stations.csv does not match the data. missing: {sorted(missing)}, extra: {sorted(extra)}")
    return stations


def nearest_grid(stations, lats, lons):
    """Pick the ERA5 grid point closest to each station.

    What: adds grid_lat, grid_lon and distance_km (station to grid point).
    How:  the ERA5 grid is regular (every 0.25°), so the closest point is the
          closest latitude combined with the closest longitude. The distance uses
          111 km per degree, with longitude degrees shrunk by cos(latitude).
    """
    out = stations.copy()
    out["grid_lat"] = [lats[np.abs(lats - v).argmin()] for v in out["latitude"]]
    out["grid_lon"] = [lons[np.abs(lons - v).argmin()] for v in out["longitude"]]
    dy = (out["latitude"] - out["grid_lat"]) * 111.0
    dx = (out["longitude"] - out["grid_lon"]) * 111.0 * np.cos(np.radians(out["latitude"]))
    out["distance_km"] = np.hypot(dx, dy).round(1)
    return out


def read_station_wind(files, grid):
    """Hourly ERA5 u10 and v10 at each station's grid point, all months stacked.

    What: returns a long table: time (UTC), station, u10, v10 (m/s).
    How:  opens one month file at a time, keeps ONLY u10 and v10 (every other
          ERA5 variable is dropped before anything is loaded), and picks the
          station grid points with xarray .sel on a "station" axis (stations
          sharing a grid point get the same numbers). The time axis is
          "valid_time" in new CDS files and "time" in older ones; both work.
    """
    lat = xr.DataArray(grid["grid_lat"].values, dims="station", coords={"station": grid["station"].values})
    lon = xr.DataArray(grid["grid_lon"].values, dims="station", coords={"station": grid["station"].values})
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = ds[WIND].drop_vars(["expver", "number"], errors="ignore")
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            points = ds.sel(latitude=lat, longitude=lon).load()
        table = points.to_dataframe().reset_index()
        parts.append(table.drop(columns=["latitude", "longitude", "expver", "number"], errors="ignore"))
    return pd.concat(parts, ignore_index=True)


def to_ist_hour(utc_times):
    """Turn ERA5 UTC times into the CPCB hour (Indian time) that contains them.

    What: 00:00 UTC -> 05:30 IST -> CPCB hour 05:00 (the 05:00-06:00 hour).
    How:  add 5 h 30 min, then floor to the full hour. CPCB rows are labelled by
          their start ("From Date"), so the floored time is the matching label.
    """
    return (utc_times + IST_OFFSET).dt.floor("h")


def add_net_speed(wind):
    """Net wind speed from the two wind parts, then the era5_ prefix.

    What: adds ws = net wind speed (m/s); u10 (west->east part) and v10
          (south->north part) are kept. Columns become era5_u10, era5_v10, era5_ws.
    How:  the two parts are at right angles, so the net speed is the length of
          the arrow they make: sqrt(u² + v²) (Pythagoras).
    """
    out = wind.copy()
    out["ws"] = np.hypot(wind["u10"], wind["v10"])
    return out.rename(columns={c: f"era5_{c}" for c in ["u10", "v10", "ws"]})


def main():
    """Run step 2.7: build merged.csv with ERA5 wind only and print the check.

    What: picks each station's nearest ERA5 grid point, converts times to IST
          CPCB hours, adds the net wind speed, left-joins onto cpcb_clean (every
          CPCB row kept), saves processed_dir/merged.csv (CSV only), and prints
          row count, grid points used, missing ERA5 rows and a wind summary.
    How:  all file names come from config.yaml; cpcb_clean.csv and the ERA5 files
          are only read, never changed. No ERA5 variable other than u10, v10 is read.
    """
    cfg = load_config()
    clean = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_clean_file"], parse_dates=["time"])
    files = sorted(Path(cfg["era5_dir"]).glob("era5_*.nc"))
    if not files:
        raise SystemExit(f"no ERA5 files in {cfg['era5_dir']}")

    stations = load_stations(cfg["stations_file"], clean["station"].unique())
    with xr.open_dataset(files[0]) as ds:
        grid = nearest_grid(stations, ds["latitude"].values, ds["longitude"].values)

    wind = read_station_wind(files, grid)
    wind["time"] = to_ist_hour(wind["time"]).astype(clean["time"].dtype)
    wind = add_net_speed(wind)

    merged = clean.merge(wind, on=["station", "time"], how="left", validate="one_to_one")
    out = cfg["processed_dir"] / cfg["merged_file"]
    merged.to_csv(out, index=False)

    era5_cols = [c for c in merged.columns if c.startswith("era5_")]
    n_grid = grid[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]
    no_era5 = merged[merged["era5_ws"].isna()].groupby("station").size()
    unused = len(wind) - merged["era5_ws"].notna().sum()
    summary = merged.groupby("station")["era5_ws"].describe()[["mean", "min", "max"]].round(2)

    pd.set_option("display.width", 200)
    lines = [
        f"ERA5 files read: {len(files)}  (variables read: {WIND})",
        f"rows: cpcb_clean {len(clean)}  merged {len(merged)}",
        f"ERA5 columns added: {era5_cols}",
        f"different grid points used by the {len(grid)} stations: {n_grid}",
        "\nstation -> nearest ERA5 grid point:",
        grid.to_string(index=False),
        "\nCPCB rows with no ERA5 value, per station:",
        no_era5.to_string() if len(no_era5) else "none",
        f"ERA5 rows outside the CPCB hours (not used): {unused}",
        "\nnet wind speed era5_ws (m/s) per station:",
        summary.to_string(),
        f"\nsaved to: {out}  shape {merged.shape}",
    ]
    text = "\n".join(lines)
    print(text)

    save_text(cfg, "2.7_merge_report.txt", text)
    save_table(cfg, "2.7_grid_points.csv", grid.set_index("station"))
    save_table(cfg, "2.7_wind_summary.csv", summary)
    print(f"temp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
