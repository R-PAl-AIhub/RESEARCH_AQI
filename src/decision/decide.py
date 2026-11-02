"""Step 7.1 (Stage 5) for PROJECT_2: per sample, target and horizon - forecast, range and flag on test 2025.

Groups = the 4 seasons (PROJECT_2 clusters). Worker B flags come from the inputs only, so one flag
per sample and horizon serves both targets.
"""
import json

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import load, short
from src.temp_out import save_table, save_text
from src.worker_b.distance import group_of, load_shapes, worker_dir
from src.worker_b.labels import LABELS, label_split
from src.worker_b.raw_score import raw_features
from src.worker_c.methods import adaptive_ranges, level_score, stored_scores


def worker_b_test(cfg, model, test):
    """Run Worker B (step 5.6b) on test, ranked against train 2023-24 (same season), and save it.

    What: returns the label table and saves worker_b/worker_b_test.csv.
    How:  label_split against the train rows of the same season (s_embed and s_raw); raw features =
          the step 5.6 columns of the latest input hour.
    """
    w, samples_dir = cfg["worker_b"], cfg["samples_dir"]
    names = json.loads((samples_dir / "feature_names.json").read_text(encoding="utf-8"))
    saved = np.load(cfg["forecasts_dir"] / w["raw_shapes_file"])
    keep = [names.index(n) for n in saved["features"]]
    e_mus, e_covs, _ = load_shapes(worker_dir(cfg, model) / w["shapes_file"])
    shapes = ((e_mus, e_covs), (saved["mu"], saved["cov"]))
    train = load(model, "train")
    labels = label_split(test, raw_features(test, samples_dir, "test", keep),
                         train, raw_features(train, samples_dir, "train", keep), shapes, cfg)
    labels.to_csv(worker_dir(cfg, model) / "worker_b_test.csv", index=False)
    return labels


def decide(test, labels, scores, cfg):
    """Forecast, range and flag per sample, target and horizon (Stage 5 decision).

    What: returns (decisions, level flags). Per horizon h: flag_h<h>; per key
          <t>_h<h>: forecast_, lower_, upper_ (public; empty for Red),
          base_lower_/base_upper_ (Green range) and yellow_lower_/yellow_upper_
          (stored for every sample, also Red, for step 7.2).
    How:  base = Mondrian CQR by season at the adaptive working level (step 6.4);
          the update counts misses against the base range for every sample,
          whatever its flag. Yellow = same season at factor x working level.
          Green -> base range, Yellow -> Yellow range, Red -> abstain.
    """
    alpha, gamma = cfg["alpha"], cfg["worker_c"]["gamma"]
    factor = cfg["decision"]["yellow_level_factor"]
    groups, times = group_of(test, cfg["worker_c"]["group_by"]), test["timestamp"].to_numpy()   # Worker C: one group
    out = test[["station", "timestamp", "season"]].copy()
    left = {"below 0": 0, "above 1": 0}
    for h in cfg["horizons"]:
        flag = labels[f"label_h{h}"].to_numpy()
        out[f"flag_h{h}"] = flag
        red, yellow = flag == "Red", flag == "Yellow"
        for target in cfg["targets"]:
            key = f"{short(target)}_h{h}"
            b_low, b_high, trace = adaptive_ranges(test, groups, scores, key, h, alpha, gamma)
            for name in left:
                left[name] += int((trace["flag"] == name).sum())
            trace["Q_yellow"] = [level_score(scores[(g, key)], lv * factor)[0]
                                 for g, lv in zip(trace["group"], trace["level"])]
            q_y = trace.set_index(["group", "time"])["Q_yellow"].reindex(
                pd.MultiIndex.from_arrays([groups, times])).to_numpy()
            y_low, y_high = test[f"low_{key}"].to_numpy() - q_y, test[f"high_{key}"].to_numpy() + q_y
            out[f"forecast_{key}"] = np.where(red, np.nan, test[f"median_{key}"].to_numpy())
            out[f"lower_{key}"] = np.where(red, np.nan, np.where(yellow, y_low, b_low))
            out[f"upper_{key}"] = np.where(red, np.nan, np.where(yellow, y_high, b_high))
            out[f"base_lower_{key}"], out[f"base_upper_{key}"] = b_low, b_high
            out[f"yellow_lower_{key}"], out[f"yellow_upper_{key}"] = y_low, y_high
    return out, left


def flag_table(decisions, targets, horizons):
    """Share of each flag and average width per flag, per target and horizon.

    What: one row per (target, horizon, flag): share, samples, average public,
          base and Yellow width of those samples.
    How:  public width = upper - lower (empty for Red).
    """
    rows = []
    for target in targets:
        for h in horizons:
            key = f"{short(target)}_h{h}"
            for flag in LABELS:
                m = decisions[f"flag_h{h}"] == flag
                d = decisions[m]
                rows.append({"target": target, "horizon": h, "flag": flag, "share": m.mean(), "samples": int(m.sum()),
                             "public width": (d[f"upper_{key}"] - d[f"lower_{key}"]).mean(),
                             "base width": (d[f"base_upper_{key}"] - d[f"base_lower_{key}"]).mean(),
                             "yellow width": (d[f"yellow_upper_{key}"] - d[f"yellow_lower_{key}"]).mean()})
    return pd.DataFrame(rows).round(3)


def main():
    """Run step 7.1 for every forecaster on test: Worker B labels, then the decision table.

    What: per model saves worker_b/worker_b_test.csv and <model>/decisions_test.csv;
          prints the share of each flag and the average width per flag. Settings
          (alpha, gamma, Yellow factor, cut-points) were fixed in config.yaml before this ran.
    How:  stored scores = CQR scores of all calibration samples per season;
          adaptive update walks the test period in time order, an outcome used
          only once known (t + h). Copies to temp/.
    """
    cfg = load_config()
    targets, horizons = cfg["targets"], cfg["horizons"]
    keys = [f"{short(t)}_h{h}" for t in targets for h in horizons]
    lines = [f"alpha {cfg['alpha']}, gamma {cfg['worker_c']['gamma']}, "
             f"Yellow = {cfg['decision']['yellow_level_factor']} x working level, Worker B groups = {cfg['worker_b']['group_by']}, Worker C groups = {cfg['worker_c']['group_by']}"]
    for model in cfg["forecasters"]["names"]:
        calib, test = load(model, "calibration"), load(model, "test")
        labels = worker_b_test(cfg, model, test)
        scores = stored_scores(calib, group_of(calib, cfg["worker_c"]["group_by"]), keys)
        decisions, left = decide(test, labels, scores, cfg)
        decisions.to_csv(cfg["forecasts_dir"] / model / cfg["evaluation"]["flags_file"], index=False)
        table = flag_table(decisions, targets, horizons)
        save_table(cfg, f"7.1_{model}_flag_widths.csv", table)
        lines += [f"\n===== {model} =====  {len(decisions)} test samples; working level left 0-1: {left}",
                  table.pivot_table(index=["target", "horizon"], columns="flag",
                                    values=["share", "public width"], sort=False).round(3).to_string()]
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "7.1_decisions_report.txt", text)


if __name__ == "__main__":
    main()
