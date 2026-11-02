"""Read one CPCB 'RealTimeReport' Excel file into a tidy hourly table."""
import sys
import warnings

import pandas as pd

from src.config import load_config
from src.temp_out import save_table

HEADER_KEY = "From Date"
TIME_FORMAT = "%d-%m-%Y %H:%M"
MAX_HEADER_SEARCH = 50  # rows scanned for the header before giving up


def _find_header_row(raw):
    first_col = raw.iloc[:MAX_HEADER_SEARCH, 0].astype(str).str.strip()
    hits = first_col.index[first_col == HEADER_KEY]
    if len(hits) == 0:
        raise ValueError(f"no '{HEADER_KEY}' cell in the first {MAX_HEADER_SEARCH} rows")
    return hits[0]


def _read_metadata(raw, header_row):
    """Key/value pairs above the header.

    What: column 0 = key, value = first non-empty cell to its right.
    How:  the key sits in merged cells A:B, so column 1 is empty and the value
          is in column 2; taking the first non-empty cell works for both layouts.
    """
    meta = {}
    for row in raw.iloc[:header_row].itertuples(index=False):
        key, values = row[0], [v for v in row[1:] if pd.notna(v)]
        if pd.notna(key) and values:
            meta[str(key).strip()] = str(values[0]).strip()
    return {
        "station": meta["Station"],
        "from_date": pd.Timestamp(meta["From"]),
        "to_date": pd.Timestamp(meta["To"]),
        "avg_period": meta.get("AvgPeriod"),
    }


def read_one(path):
    """Return (table, meta) for one CPCB Excel file.

    table: columns station, time, then every variable column (all numeric;
           unreadable cells become NaN). `time` is the parsed "From Date".
    meta:  station name, From and To dates, averaging period, as written in the
           file's metadata rows.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        raw = pd.read_excel(path, header=None, dtype=object, engine="openpyxl")

    header_row = _find_header_row(raw)
    meta = _read_metadata(raw, header_row)

    body = raw.iloc[header_row + 1:].copy()
    body.columns = [str(c).strip() for c in raw.iloc[header_row]]
    body = body.dropna(how="all")

    # strict parse: a malformed timestamp is a format change we want to see, not hide
    time = pd.to_datetime(body["From Date"].astype(str).str.strip(), format=TIME_FORMAT)

    variables = [c for c in body.columns if c not in ("From Date", "To Date")]
    values = body[variables].apply(pd.to_numeric, errors="coerce").astype("float64")

    table = pd.concat(
        [pd.DataFrame({"station": meta["station"], "time": time.to_numpy()}),
         values.reset_index(drop=True)],
        axis=1,
    )
    return table, meta


def _check(path):
    """Print the step 2.1 check for one file and save its table as CSV.

    What: reads the file with read_one, prints shape, time range, types and
          missing counts, and writes the table to temp/2.1_one_file.csv.
    How:  temp_dir comes from config.yaml; save_table writes a CSV (no parquet)
          and replaces an older copy from a previous run.
    """
    table, meta = read_one(path)
    out = save_table(load_config(), "2.1_one_file.csv", table.set_index("station"))
    print(f"file:      {path}")
    print(f"metadata:  station={meta['station']!r}  From={meta['from_date'].date()}  "
          f"To={meta['to_date'].date()}  AvgPeriod={meta['avg_period']}")
    print(f"shape:     {table.shape}")
    print(f"columns:   {list(table.columns)}")
    print(f"first:     {table['time'].iloc[0]}")
    print(f"last:      {table['time'].iloc[-1]}")
    print(f"dtypes:    station={table['station'].dtype}, time={table['time'].dtype}, "
          f"variables all float64 = {all(table[c].dtype == 'float64' for c in table.columns[2:])}")
    print("missing per column:")
    print(table.isna().sum().to_string())
    print(f"saved:     {out}")


if __name__ == "__main__":
    _check(sys.argv[1])
