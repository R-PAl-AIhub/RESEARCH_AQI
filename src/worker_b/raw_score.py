"""Step 5.6 (Worker B safety net): distance on the raw input features -> s_raw, fused with s_embed.

PROJECT_2: the raw shapes are fitted per season on train, so they do not depend on the forecaster
and are fitted once; s_raw is ranked against the train hours of the same season (like s_embed);
s_embed differs per forecaster.
"""
import json

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import load
from src.splits.weeks import alternating_weeks
from src.temp_out import save_table, save_text
from src.worker_b.distance import fit_shapes, group_of, group_order, load_shapes, own_distance, worker_dir
from src.worker_b.pvalue import conformal_pvalues, embed_score

SAMPLE_FILES = {"train": "train", "validation": "val", "calibration": "calib", "test": "test"}   # export -> step 3.3 name


def raw_columns(names, drop_columns):
    """Which of the step 3.3 feature columns go into the raw distance.

    What: returns (kept indexes, kept names, dropped names).
    How:  drops every name in `drop_columns` (BP, stuck_pm25, stuck_pm10).
    """
    keep = [i for i, n in enumerate(names) if n not in drop_columns]
    dropped = [n for n in names if n in drop_columns]
    return keep, [names[i] for i in keep], dropped


def raw_features(table, samples_dir, split, keep):
    """Scaled raw features of the latest input hour, in the row order of an exported table.

    What: returns an array (rows of `table`, len(keep)).
    How:  X_<split>.npy is opened without reading it all (mmap) and only the last
          of the 48 hours is taken. Its rows are matched to `table` by station and
          timestamp (= last input hour in both files); every row must match once.
    """
    sample_file = SAMPLE_FILES[split]
    X = np.load(samples_dir / f"X_{sample_file}.npy", mmap_mode="r")
    last_hour = np.asarray(X[:, -1, :])[:, keep]
    meta = pd.read_csv(samples_dir / f"meta_{sample_file}.csv", usecols=["station", "timestamp"],
                       parse_dates=["timestamp"])
    meta["row"] = np.arange(len(meta))
    matched = table[["station", "timestamp"]].merge(meta, on=["station", "timestamp"], how="left",
                                                    validate="one_to_one")
    if matched["row"].isna().any():
        raise SystemExit(f"{matched['row'].isna().sum()} exported samples have no row in X_{sample_file}.npy")
    return last_hour[matched["row"].to_numpy()]


def raw_score(features, seasons, ref_features, ref_seasons, mus, covs, order):
    """s_raw = 1 - seasonal p-value of the raw distance (any reference set).

    What: returns (s_raw, raw distance) for every row of `features`.
    How:  raw distance = Mahalanobis distance to the row's own season shape
          (step 5.2 function); p-value = step 5.3 function against the
          reference rows of the same season; s_raw = 1 - p.
    """
    d = own_distance(features, seasons, mus, covs, order)
    ref_d = own_distance(ref_features, ref_seasons, mus, covs, order)
    return embed_score(conformal_pvalues(ref_d, ref_seasons, d, seasons)), d


def fused_score(s_raw, s_embed):
    """Fused novelty score = max(s_raw, s_embed).

    What: returns the larger of the two scores per sample.
    How:  if either view (raw inputs or the model's notepad) finds the hour
          unusual, the fused score is high; the notepad alone can call a wild
          input "normal", the raw inputs cannot.
    """
    return np.maximum(s_raw, s_embed)


def main():
    """Run step 5.6: fit raw-feature shapes on TRAIN once, then compare s_raw with s_embed per forecaster.

    What: saves forecasts_dir/raw_shapes.npz (mu, cov, condition numbers,
          feature names, seasons). Per model saves worker_b/raw_check.csv
          (check-week station, timestamp, season, s_raw, s_embed, fused) and
          prints the correlation of s_raw with s_embed and the share where s_raw
          is the larger one. The test split is not read. Copies go to temp/.
    How:  shapes by step 5.2 fit_shapes on the train rows (any forecaster's train
          table gives the same samples); s_embed as in step 5.3 (against the
          train hours); s_raw against the train hours of the same season; reported on the
          even (check) weeks of calibration.
    """
    cfg = load_config()
    w_cfg, samples_dir = cfg["worker_b"], cfg["samples_dir"]
    group_by = w_cfg["group_by"]
    order = group_order(cfg, group_by)
    names = json.loads((samples_dir / "feature_names.json").read_text(encoding="utf-8"))
    keep, kept_names, dropped = raw_columns(names, w_cfg["raw_drop_columns"])
    models = cfg["forecasters"]["names"]

    first_train, first_calib = load(models[0], "train"), load(models[0], "calibration")
    train_raw = raw_features(first_train, samples_dir, "train", keep)
    calib_raw = raw_features(first_calib, samples_dir, "calibration", keep)
    first_groups = group_of(first_train, group_by)
    mus, covs, report = fit_shapes(train_raw, first_groups, order, w_cfg)
    raw_file = cfg["forecasts_dir"] / w_cfg["raw_shapes_file"]
    np.savez(raw_file, mu=mus, cov=covs, seasons=np.array(order), features=np.array(kept_names),
             condition=report["condition"].to_numpy())

    chk = (alternating_weeks(first_calib["timestamp"]) == "check").to_numpy()
    seasons = group_of(first_calib, group_by)
    s_raw, _ = raw_score(calib_raw[chk], seasons[chk], train_raw, first_groups, mus, covs, order)

    lines = [f"raw features: {len(kept_names)} of {len(names)}   dropped: {dropped}",
             f"\nraw covariance per group {order} (plain mean + covariance of train, limit {w_cfg['max_condition']}):",
             report.to_string(),
             f"\ns_raw ranked against {len(train_raw)} train hours of the same season; "
             f"reported on the even (check) weeks of calibration: {chk.sum()} samples"]
    rows = []
    for model in models:
        calib = load(model, "calibration")
        assert calib[["station", "timestamp"]].equals(first_calib[["station", "timestamp"]])
        e_mus, e_covs, _ = load_shapes(worker_dir(cfg, model) / w_cfg["shapes_file"])
        train = load(model, "train")
        train_groups = group_of(train, group_by)
        d_emb = own_distance(calib, seasons, e_mus, e_covs, order)
        d_train = own_distance(train, train_groups, e_mus, e_covs, order)
        s_embed = embed_score(conformal_pvalues(d_train, train_groups, d_emb[chk], seasons[chk]))
        fused = fused_score(s_raw, s_embed)
        calib.loc[chk, ["station", "timestamp", "season"]].assign(s_raw=s_raw, s_embed=s_embed, fused=fused) \
            .to_csv(worker_dir(cfg, model) / "raw_check.csv", index=False)
        rows.append({"model": model, "pearson": np.corrcoef(s_raw, s_embed)[0, 1],
                     "spearman": pd.Series(s_raw).corr(pd.Series(s_embed), method="spearman"),
                     "s_raw larger": (s_raw > s_embed).mean(), "s_embed larger": (s_embed > s_raw).mean(),
                     "s_embed > 0.98": (s_embed > 0.98).mean(), "s_raw > 0.98": (s_raw > 0.98).mean(),
                     "fused > 0.98": (fused > 0.98).mean()})
    compare = pd.DataFrame(rows).set_index("model").round(3)
    lines += ["share of check-week samples and correlation of s_raw with s_embed:", compare.to_string(),
              f"\nsaved: {raw_file}"]
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "5.6_raw_score_report.txt", text)
    save_table(cfg, "5.6_raw_shapes.csv", report)
    save_table(cfg, "5.6_score_compare.csv", compare)


if __name__ == "__main__":
    main()
