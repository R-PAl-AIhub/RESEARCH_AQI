"""Step 3.2: fill empty input cells (one imputation method per run) and build the model's input columns."""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes IterativeImputer importable)
from sklearn.impute import IterativeImputer, KNNImputer

from src.config import load_config
from src.temp_out import save_table, save_text

ID_COLUMNS = ("station", "time", "row_present")
NOT_FEATURES = ("station", "time", "row_present", "season")   # plus every target_ column


def target_column(target):
    """Name of the unfilled copy of one target: "PM2.5" -> "target_pm25", "PM10" -> "target_pm10".

    What: returns the column name used for that target in features and samples.
    How:  lower case, dot removed, "target_" in front.
    """
    return "target_" + target.lower().replace(".", "")


def input_columns(df):
    """The input variables: every kept CPCB variable and every era5_ column.

    What: returns the column names that get gap-filled and fed to the model
          (AOD and AOD_present too, if step 2.8 added them).
    How:  everything except the ID columns and the flag_ columns (the flags only
          explain why a value was blanked in step 2.5; they are not inputs).
    """
    return [c for c in df.columns if c not in ID_COLUMNS and not c.startswith("flag_")]


def impute(df, columns, imp):
    """Fill empty input cells with the run's method (the target_ copies are not in `columns`).

    What: returns a copy. imp["method"]: deletion = nothing filled; ffill / bfill =
          last earlier / next later value of the same station, no time limit;
          median = train median of the column per station; knn = mean of the k
          most similar train hours; mice = iterative imputation fitted on train.
          Cells that had a value keep it exactly.
    How:  median, knn and mice learn only from rows up to imp["train_end"]. For
          them wind direction is filled as sin/cos and turned back into degrees
          (the mean of 350 and 10 degrees is 0, not 180).
    """
    method, out = imp["method"], df.copy()
    if method == "deletion":
        return out
    if method in ("ffill", "bfill"):
        groups = df.groupby("station")[columns]
        out[columns] = groups.ffill() if method == "ffill" else groups.bfill()
        return out
    x = wd_to_circle(df[columns])
    train = (df["time"] <= pd.Timestamp(imp["train_end"])).to_numpy()
    if method == "median":
        median = x[train].groupby(df.loc[train, "station"]).median()
        x = x.fillna(median.reindex(df["station"]).set_index(x.index))
    else:
        x = model_impute(x, train, imp)
    filled = wd_to_degrees(x)[columns]
    out[columns] = df[columns].where(df[columns].notna(), filled)
    return out


def wd_to_circle(x):
    """WD (degrees) -> WD_sin, WD_cos, so averages and models respect the circle."""
    if "WD" not in x.columns:
        return x.copy()
    rad = np.radians(x["WD"])
    return x.drop(columns="WD").assign(WD_sin=np.sin(rad), WD_cos=np.cos(rad))


def wd_to_degrees(x):
    """WD_sin, WD_cos -> WD in degrees 0..360 (inverse of wd_to_circle)."""
    if "WD_sin" not in x.columns:
        return x
    wd = np.degrees(np.arctan2(x["WD_sin"], x["WD_cos"])) % 360
    return x.drop(columns=["WD_sin", "WD_cos"]).assign(WD=wd)


def model_impute(x, train, imp):
    """KNN or MICE imputation, learned on the train rows only.

    What: returns x with every empty cell filled.
    How:  columns are put on one scale with the train mean and spread (so no
          column dominates the distances), the imputer is fitted on train rows
          (KNN: a seeded sample of imp["knn_donors"] train rows as donors), only
          rows with an empty cell are transformed, then the scale is undone.
    """
    mean, std = x[train].mean(), x[train].std().replace(0, 1)
    z = (x - mean) / std
    if imp["method"] == "knn":
        donors = z[train].sample(min(imp["knn_donors"], int(train.sum())), random_state=imp["seed"])
        model = KNNImputer(n_neighbors=imp["knn_neighbors"]).fit(donors)
    elif imp["method"] == "mice":
        model = IterativeImputer(max_iter=imp["mice_iterations"], random_state=imp["seed"]).fit(z[train])
    else:
        raise ValueError(f"unknown imputation method: {imp['method']}")
    rows = z.isna().any(axis=1).to_numpy()
    if rows.any():
        z.loc[rows] = model.transform(z[rows])
    return z * std + mean


def longest_filled(before, after, columns):
    """Per input variable: how many cells were filled and the longest filled run.

    What: one row per variable: cells_filled, longest_run_hours.
    How:  a cell was filled if it was missing before and has a value after. A run
          is a block of consecutive filled hours inside one station; a new block
          starts whenever "filled" switches or the station changes.
    """
    rows = {}
    new_station = before["station"] != before["station"].shift()
    for col in columns:
        filled = before[col].isna() & after[col].notna()
        block = ((filled != filled.shift()) | new_station).cumsum()
        runs = filled.groupby(block).sum()
        rows[col] = {"cells_filled": int(filled.sum()), "longest_run_hours": int(runs.max())}
    table = pd.DataFrame(rows).T
    table.index.name = "variable"
    return table


def add_lags(df, variables, lags):
    """Add past values: <var>_lag<h> = the value h hours earlier at the same station.

    What: one new column per variable and lag.
    How:  rows of one station are consecutive hours (step 2.2 grid), so
          groupby(station).shift(h) moves values h rows (= h hours) down.
    """
    for var in variables:
        for lag in lags:
            df[f"{var}_lag{lag}"] = df.groupby("station")[var].shift(lag)
    return df


def add_rolling(df, var, hours, stat):
    """Add <var>_<stat><h>h = mean or standard deviation of the last `hours` hours, this hour included.

    What: one new column, e.g. PM2.5_mean24h or PM2.5_std12h. At 10:00 a 12 h
          window covers 23:00 the day before to 10:00 today (past only).
    How:  a rolling window per station that only looks backwards; missing hours
          are skipped. The mean needs 1 value, the std needs 2 (else missing).
    """
    roll = lambda s: s.rolling(hours, min_periods=1 if stat == "mean" else 2)
    df[f"{var}_{stat}{hours}h"] = df.groupby("station")[var].transform(
        lambda s: roll(s).mean() if stat == "mean" else roll(s).std())
    return df


def add_cyclic(df):
    """Turn circular quantities into sine and cosine pairs.

    What: WD -> WD_sin, WD_cos (WD removed); adds hour_sin, hour_cos, doy_sin, doy_cos.
    How:  a circle has no start, so 359° sits next to 1° and 23:00 next to 00:00.
          sin and cos of the angle keep those neighbours close; the raw numbers
          would put them far apart. Day of year uses 365.25 days per circle.
    """
    wd = np.radians(df["WD"])
    df["WD_sin"], df["WD_cos"] = np.sin(wd), np.cos(wd)
    df = df.drop(columns="WD")
    hour = 2 * np.pi * df["time"].dt.hour / 24
    doy = 2 * np.pi * (df["time"].dt.dayofyear - 1) / 365.25
    df["hour_sin"], df["hour_cos"] = np.sin(hour), np.cos(hour)
    df["doy_sin"], df["doy_cos"] = np.sin(doy), np.cos(doy)
    return df


def build_features(merged, feat_cfg, targets, imp):
    """Build the feature table from the merged table (used by main and by the test).

    What: returns (features, fill_report). features has station, time,
          row_present, one unfilled target_ column per target (target_pm25,
          target_pm10), season and the input columns.
    How:  sort by station and time, keep an unfilled copy of each target, fill
          empty input cells with impute() (method in imp), then add lags, the rolling means and stds, the sine/cosine pairs and the
          season (month -> label from config features.seasons: winter,
          summer_premonsoon, sw_monsoon, post_monsoon). Every step only looks at
          the same hour or earlier hours.
    """
    df = merged.sort_values(["station", "time"]).reset_index(drop=True)
    inputs = input_columns(df)
    for target in targets:
        df[target_column(target)] = df[target]

    filled = impute(df, inputs, imp)
    report = longest_filled(df, filled, inputs)

    out = filled.drop(columns=[c for c in filled.columns if c.startswith("flag_")])
    out = add_lags(out, feat_cfg["lag_variables"], feat_cfg["lags"])
    for var in feat_cfg["rolling_variables"]:
        for hours in feat_cfg["rolling_hours"]:
            for stat in feat_cfg["rolling_stats"]:
                out = add_rolling(out, var, hours, stat)
    out = add_cyclic(out)
    out["season"] = out["time"].dt.month.map(feat_cfg["seasons"])
    return out, report


def feature_names(features):
    """The model input columns, in table order.

    What: every column except station, time, row_present, season and the target_ columns.
    How:  a simple filter on the column names.
    """
    return [c for c in features.columns if c not in NOT_FEATURES and not c.startswith("target_")]


def main():
    """Run step 3.2: build features.csv (CSV only) and print the check.

    What: prints the feature list, how many cells each variable got filled,
          the longest filled run, and whether AOD was used. Copies go to temp/.
    How:  reads cpcb_clean.csv (never changes it; step 2.7 ERA5 wind removed);
          settings come from the features section of config.yaml.
    """
    cfg = load_config()
    merged = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_clean_file"], parse_dates=["time"])
    imp = {**cfg["imputation"], "train_end": cfg["splits"]["train_end"], "seed": cfg["seed"]}
    features, report = build_features(merged, cfg["features"], cfg["targets"], imp)

    out = cfg["processed_dir"] / cfg["features_file"]
    features.to_csv(out, index=False)

    names = feature_names(features)
    lines = [
        f"AOD used: {'yes' if 'AOD' in features.columns else 'no (step 2.8 not done)'}",
        f"\n{len(names)} features:",
        ", ".join(names),
        "\nfilled cells and longest filled run per input variable:",
        report.to_string(),
        f"\nimputation method: {imp['method']}; longest filled run: {report['longest_run_hours'].max()} hours",
        *[f"{target_column(t)} missing (unfilled): {features[target_column(t)].isna().sum()} of {len(features)}"
          for t in cfg["targets"]],
        f"\nsaved to: {out}  shape {features.shape}",
    ]
    text = "\n".join(lines)
    print(text)

    save_text(cfg, "3.2_features_report.txt", text)
    save_table(cfg, "3.2_fill_report.csv", report)
    print(f"temp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
