"""Step 3.1 (GATE): show data per split and season, and check the group-size rules for d and the levels."""
import math

import pandas as pd

from src.config import load_config
from src.temp_out import save_table, save_text

D_VALUES = (6, 8, 16, 32)       # embedding sizes to test (6 = config d with seasons as clusters)
RED_LEVEL = 0.02                # Worker B Red level (must pass)
ALPHAS = (0.10, 0.05)           # Worker C levels
ALL_ROW = "ALL (calendar dates)"


def format_shares(df, variables, years):
    """Share of exact zeros and of whole numbers, per variable and year.

    What: returns two tables (variables x years), in % of the non-missing values.
          (Copied unchanged from PROJECT src/clean/apply.py, which PROJECT_2 does not have.)
    How:  a value is "whole" if rounding it does not change it (12.0 yes, 12.25
          no). If one year stores rounded numbers and another stores decimals,
          the whole-number share jumps between those years: a reporting-format
          change, not a change in the air. Missing values count in neither part.
    """
    values = df[variables]
    present = values.notna().groupby(years).sum()
    zeros = (values == 0).groupby(years).sum() / present * 100
    whole = (values == values.round()).groupby(years).sum() / present * 100
    return zeros.T.round(1), whole.T.round(1)


PARTS = ("train", "val", "calib", "test")


def split_options():
    """The split option chosen on 2026-10-16 (option C), as date ranges.

    What: returns {name: {part: (first hour, last hour)}} for train, val, calib, test.
    How:  option C: train = all of 2023 and 2024; 2025 in three 4-month blocks:
          Jan-Apr validation, May-Aug calibration, Sep-Dec test (adaptive range).
          Ends are included (IST hour starts).
    """
    return {"C": {"train": ("2023-01-01 00:00", "2024-12-31 23:00"),
                  "val": ("2025-01-01 00:00", "2025-04-30 23:00"),
                  "calib": ("2025-05-01 00:00", "2025-08-31 23:00"),
                  "test": ("2025-09-01 00:00", "2025-12-31 23:00")}}


def target_hours(merged, targets, seasons):
    """Only the hours where EVERY target (PM2.5 and PM10) has a value, with date and season added.

    What: returns station, time, year, date, season for every such hour.
    How:  drops rows where any target is missing (a sample needs all targets);
          the date is the time without the hour; the season comes from the month
          through `seasons` (config features.seasons: winter, summer_premonsoon,
          sw_monsoon, post_monsoon).
    """
    df = merged.loc[merged[targets].notna().all(axis=1), ["station", "time"]].copy()
    df["year"] = df["time"].dt.year
    df["date"] = df["time"].dt.normalize()
    df["season"] = df["time"].dt.month.map(seasons)
    return df


def count_part(df, span, season_order):
    """Hours and distinct days per station and season inside one split part.

    What: returns two tables (hours, days): rows = stations plus one ALL row,
          columns = the 4 seasons plus "total" (seasons shown for information).
    How:  span = (first hour, last hour), both included. Hours = number of rows.
          Days = number of different dates. In the ALL row a date counts once even
          if 7 stations have data on it, because the stations share nearly the
          same weather and are not independent days.
    """
    part = df[(df["time"] >= pd.Timestamp(span[0])) & (df["time"] <= pd.Timestamp(span[1]))]
    hours, days = {}, {}
    groups = [(name, g) for name, g in part.groupby("station")] + [(ALL_ROW, part)]
    for name, g in groups:
        by_season = g.groupby("season")
        hours[name] = {**by_season.size().to_dict(), "total": len(g)}
        days[name] = {**by_season["date"].nunique().to_dict(), "total": g["date"].nunique()}
    columns = season_order + ["total"]
    hours = pd.DataFrame(hours).T.reindex(columns=columns).fillna(0).astype(int)
    days = pd.DataFrame(days).T.reindex(columns=columns).fillna(0).astype(int)
    hours.index.name = days.index.name = "station"
    return hours, days


def min_days_for_level(level):
    """Fewest calibration days a group needs so a p-value can reach `level`.

    What: returns 1/level - 1, rounded up (0.02 -> 49, 0.05 -> 19, 0.10 -> 9).
    How:  the smallest conformal p-value is 1/(n+1); it is <= level only when
          n >= 1/level - 1. round() removes float noise before rounding up.
    """
    return math.ceil(round(1 / level - 1, 9))


def d_rule(train_days, season_order):
    """Rule for d on TRAIN: every season group must have >= 10*d days (n >= 10*d).

    What: one row per season plus one for all hours: days, PASS/FAIL for each d,
          and the largest d that passes in that group. The d to use is the
          smallest of the season values, so the rule holds in every season.
    How:  the group sizes are the season columns and the "total" column of the
          ALL row of the train days table; largest d = days // 10.
    """
    rows = []
    for group in season_order + ["total"]:
        n = int(train_days[group])
        rows.append({"group": "all (no seasons)" if group == "total" else group, "days": n,
                     **{f"d={d} (need {10 * d})": "PASS" if n >= 10 * d else "FAIL" for d in D_VALUES},
                     "largest_d_allowed": n // 10})
    return pd.DataFrame(rows)


def level_rule(calib_days):
    """Rule for the levels on CALIBRATION: the one group >= 1/level - 1 days.

    What: one row for Worker B (Red level) and one per alpha for Worker C.
    How:  calib_days is the ALL row of the calibration days table; the group
          size is its "total" column (one group, no seasons).
    """
    n = int(calib_days["total"])
    group = "all (no seasons)"
    rows = [("B", RED_LEVEL)] + [("C", alpha) for alpha in ALPHAS]
    return pd.DataFrame([{"worker": w, "group": group, "days": n, "level": level,
                          "need": min_days_for_level(level),
                          "result": "PASS" if n >= min_days_for_level(level) else "FAIL"}
                         for w, level in rows])


def risks(df, merged, option, targets):
    """The risk list asked for in step 3.1.

    What: returns text lines: late-starting stations, 2020 lockdown inside
          training, only one training year, and the reporting change of each target.
    How:  a station "starts late" if its first date with all targets is more than
          30 days after the first such date in the file. The reporting change is
          shown as the % of values that are whole numbers, per year and target.
    """
    first = df.groupby("station")["date"].min()
    late = first[first > df["date"].min() + pd.Timedelta(days=30)]
    whole = format_shares(merged, targets, merged["time"].dt.year)[1]
    return [
        "late-starting stations: " + (", ".join(f"{s} ({d.date()})" for s, d in late.items()) or "none"),
        "2020 lockdown inside training: " + ("YES" if 2020 in option["train"] else "no (2020 not in training)"),
        "only one training year: " + ("YES " + str(option["train"]) if len(option["train"]) == 1 else "no"),
    ] + [f"reporting change, % whole-number {t} per year: " + ", ".join(f"{y}: {v}" for y, v in whole.loc[t].items())
         for t in targets]


def short_names(table):
    """Drop the common ", Kolkata - WBPCB" ending so the tables fit on screen.

    What: returns the same table with shorter row names.
    How:  string replace on the index; the data is unchanged.
    """
    return table.rename(index=lambda s: s.replace(", Kolkata - WBPCB", ""))


def main():
    """Run step 3.1: print and save the split tables, both rules and the risks.

    What: reads cpcb_clean.csv (step 2.7 ERA5 wind removed), keeps hours where every target has a value, and for
          each split option prints hours and days per station and season, the d
          rule (TRAIN) and the level rule (CALIBRATION), then the risks. Copies go
          to temp/. Nothing is written to config.yaml (GATE step).
    How:  file names, targets and seasons come from config.yaml; the data is only read.
    """
    cfg = load_config()
    targets, seasons = cfg["targets"], cfg["features"]["seasons"]
    season_order = list(dict.fromkeys(seasons.values()))   # winter, summer_premonsoon, sw_monsoon, post_monsoon
    merged = pd.read_csv(cfg["processed_dir"] / cfg["cpcb_clean_file"], parse_dates=["time"])
    df = target_hours(merged, targets, seasons)
    years = sorted(df["year"].unique().tolist())
    tag = " and ".join(targets)

    pd.set_option("display.width", 200)
    lines = [f"years with {tag}: {years}   hours with {tag}: {len(df)} of {len(merged)}"]

    for name, option in split_options().items():
        lines.append(f"\n===== option {name}: " + ", ".join(f"{p} {option[p][0]} to {option[p][1]}" for p in PARTS)
                     + " · one group (no seasons) =====")
        days_all, summary = {}, []
        for part in PARTS:
            hours, days = count_part(df, option[part], season_order)
            days_all[part] = days.loc[ALL_ROW]
            summary.append({"part": part, "hours": int(hours.loc[ALL_ROW, "total"]), "days": int(days.loc[ALL_ROW, "total"])})
            lines += [f"\n[{part}] hours with {tag}:", short_names(hours).to_string(),
                      f"\n[{part}] distinct days with {tag}:", short_names(days).to_string()]
            save_table(cfg, f"3.1_{name}_{part}_hours.csv", hours)
            save_table(cfg, f"3.1_{name}_{part}_days.csv", days)
        summary = pd.DataFrame(summary)
        summary["share_of_hours_%"] = (summary["hours"] / summary["hours"].sum() * 100).round(1)
        lines += ["\nsummary (all stations):", summary.to_string(index=False)]

        train_years = list(range(pd.Timestamp(option["train"][0]).year, pd.Timestamp(option["train"][1]).year + 1))
        d_table = d_rule(days_all["train"], season_order)
        seasons_only = d_table[d_table["group"] != "all (no seasons)"]
        level_table = level_rule(days_all["calib"])
        lines += ["\nrule for d on TRAIN (each group >= 10*d distinct days):",
                  d_table.to_string(index=False),
                  f"largest d that passes in EVERY season: {int(seasons_only['largest_d_allowed'].min())} "
                  f"(set by {seasons_only.loc[seasons_only['largest_d_allowed'].idxmin(), 'group']})",
                  "\nrule for levels on CALIBRATION (the one group >= 1/level - 1 distinct days):",
                  level_table.to_string(index=False),
                  "\nrisks:"] + ["- " + r for r in risks(df, merged, {"train": train_years}, targets)]
        save_table(cfg, f"3.1_{name}_summary.csv", summary.set_index("part"))
        save_table(cfg, f"3.1_{name}_d_rule.csv", d_table.set_index("group"))
        save_table(cfg, f"3.1_{name}_level_rule.csv", level_table)

    text = "\n".join(lines)
    print(text)
    save_text(cfg, "3.1_split_plan.txt", text)
    print(f"\ntemp copies in: {cfg['temp_dir']}")


if __name__ == "__main__":
    main()
