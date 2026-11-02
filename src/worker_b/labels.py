"""Step 5.6b (Worker B output): Green / Yellow / Red label per sample and horizon from the fused score.

PROJECT_2: no Simes cap (step 5.4 skipped), no smoothing (step 5.5 removed): Red = fused score above yellow_max.
"""
import json

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import load
from src.splits.weeks import alternating_weeks
from src.temp_out import save_table, save_text
from src.worker_b.distance import group_of, group_order, load_shapes, own_distance, season_order, worker_dir
from src.worker_b.pvalue import conformal_pvalues, embed_score
from src.worker_b.raw_score import fused_score, raw_features, raw_score

LABELS = ["Green", "Yellow", "Red"]


def scores(table, raw, train_table, train_raw, shapes, cfg):
    """s_embed, s_raw and the fused score of every row of `table`.

    What: returns (s_embed, s_raw, fused), one value per row each.
    How:  s_embed = 1 - p-value of the embedding distance against the TRAIN rows
          `train_table` (2023-24) of the same season (steps 5.2, 5.3); s_raw = the same
          on the raw features `train_raw` (step 5.6); fused = max of the two (no smoothing).
    """
    group_by = cfg["worker_b"]["group_by"]
    order = group_order(cfg, group_by)
    (e_mus, e_covs), (r_mus, r_covs) = shapes
    seasons, train_seasons = group_of(table, group_by), group_of(train_table, group_by)
    d_emb = own_distance(table, seasons, e_mus, e_covs, order)
    train_d_emb = own_distance(train_table, train_seasons, e_mus, e_covs, order)
    s_embed = embed_score(conformal_pvalues(train_d_emb, train_seasons, d_emb, seasons))
    s_raw, _ = raw_score(raw, seasons, train_raw, train_seasons, r_mus, r_covs, order)
    return s_embed, s_raw, fused_score(s_raw, s_embed)


def label_split(table, raw, train_table, train_raw, shapes, cfg):
    """Worker B output for any split: scores and a Green / Yellow / Red label per horizon.

    What: returns one row per row of `table`: station, timestamp, season (= cluster
          id), s_raw, s_embed, fused, label_h<h> per horizon.
          Both scores are ranked against `train_table` / `train_raw` (train 2023-24),
          within the row's own season (worker_b.group_by).
    How:  scores() gives the fused score of every row; the label comes straight from it:
          Green if it is <= green_max, Yellow up to yellow_max, Red above. With no smoothing
          the score is the same for every horizon, so every label_h<h> is the same; the
          per-horizon columns are kept because steps 7.1 and 7.2 read one flag per horizon.
    """
    w, horizons = cfg["worker_b"], cfg["horizons"]
    s_embed, s_raw, fused = scores(table, raw, train_table, train_raw, shapes, cfg)
    seasons = group_of(table, w["group_by"])

    out = pd.DataFrame({"station": table["station"].to_numpy(), "timestamp": table["timestamp"].to_numpy(),
                        "season": table["season"].to_numpy(), "group": seasons, "s_raw": s_raw, "s_embed": s_embed,
                        "fused": fused})
    label = np.select([fused <= w["green_max"], fused <= w["yellow_max"]], ["Green", "Yellow"], default="Red")
    for h in horizons:
        out[f"label_h{h}"] = label
    return out


def label_shares(labels, horizons, order):
    """Share of Green, Yellow and Red per horizon and season (rows "all" + 4 seasons).

    What: table with columns horizon, season, Green, Yellow, Red, n.
    How:  counts the label_h<h> values of each group and divides by the group size.
    """
    rows = []
    for h in horizons:
        df = labels[["season", f"label_h{h}"]].rename(columns={f"label_h{h}": "label"})
        for name, g in ({"all": df} | {s: df[df["season"] == s] for s in order}).items():
            share = g["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
            rows.append({"horizon": h, "season": name, **share.round(3).to_dict(), "n": len(g)})
    return pd.DataFrame(rows)


def main():
    """Run step 5.6b for every forecaster on calibration: save the labels, then the alternating-week check.

    What: per model saves worker_b/worker_b_calibration.csv (ranked against train
          2023-24, same season) and worker_b/label_check.csv (the same labels on the
          even calendar weeks), and prints the check shares of Green / Yellow / Red per
          season and horizon. Test is not read.
    How:  raw features = the step 5.6 columns (names saved in raw_shapes.npz) of
          the latest input hour; cut-points from config.yaml. Copies to temp/.
    """
    cfg = load_config()
    w, samples_dir, horizons = cfg["worker_b"], cfg["samples_dir"], cfg["horizons"]
    names = json.loads((samples_dir / "feature_names.json").read_text(encoding="utf-8"))
    saved = np.load(cfg["forecasts_dir"] / w["raw_shapes_file"])
    raw_shape = (saved["mu"], saved["cov"])
    keep = [names.index(n) for n in saved["features"]]

    lines = [f"Green s <= {w['green_max']}, Yellow <= {w['yellow_max']}, Red above (no Simes cap)"]
    for model in cfg["forecasters"]["names"]:
        out_dir = worker_dir(cfg, model)
        e_mus, e_covs, _ = load_shapes(out_dir / w["shapes_file"])
        shapes = ((e_mus, e_covs), raw_shape)
        calib = load(model, "calibration")
        calib_raw = raw_features(calib, samples_dir, "calibration", keep)
        train = load(model, "train")
        train_raw = raw_features(train, samples_dir, "train", keep)

        labels = label_split(calib, calib_raw, train, train_raw, shapes, cfg)
        labels.to_csv(out_dir / w["labels_file"], index=False)

        chk = (alternating_weeks(calib["timestamp"]) == "check").to_numpy()
        check = labels[chk]
        check.to_csv(out_dir / "label_check_rows.csv", index=False)
        present = [s for s in season_order(cfg) if s in set(check["season"])]   # report rows: seasons present
        shares = label_shares(check, horizons, present)
        shares.to_csv(out_dir / "label_check.csv", index=False)
        lines += [f"\n===== {model} =====  check weeks {chk.sum()} samples",
                  "share per label (aim: about 0.80 Green, 0.18 Yellow, 0.02 Red):",
                  shares.pivot_table(index="season", columns="horizon", values=LABELS, sort=False).to_string()]
        save_table(cfg, f"5.6b_{model}_check_shares.csv", shares)
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "5.6b_labels_report.txt", text)


if __name__ == "__main__":
    main()
