"""Step 5.3 (Worker B, blocks B5, B6, B8): conformal p-value of a distance against the train distances -> s_embed."""
import numpy as np

from src.config import load_config
from src.models.train import load
from src.temp_out import save_text
from src.worker_b.distance import group_of, group_order, load_shapes, own_distance, worker_dir

def conformal_pvalues(ref_values, ref_seasons, new_values, new_seasons):
    """p-value of each new value against the reference values of the same season.

    What: p_t = (1 + number of reference values in season S that are >= d_t) / (n_S + 1).
          Small p = rarely seen something this far. Steps 5.6 and 5.6b reuse this
          on other values. No chi-squared or other bell-curve formula is used.
    How:  for each season, sort its reference values once; searchsorted then
          gives, for every new value, how many reference values are smaller, so
          n_S minus that is the count of values >= d_t. The +1 on top and bottom
          counts the new sample itself, which makes the guarantee exact.
    """
    ref_values, ref_seasons = np.asarray(ref_values), np.asarray(ref_seasons)
    new_values, new_seasons = np.asarray(new_values), np.asarray(new_seasons)
    p = np.full(len(new_values), np.nan)
    for season in np.unique(new_seasons):
        ref = np.sort(ref_values[ref_seasons == season])
        mask = new_seasons == season
        at_least = len(ref) - np.searchsorted(ref, new_values[mask], side="left")
        p[mask] = (1 + at_least) / (len(ref) + 1)
    return p


def embed_score(p):
    """Score = 1 - p: near 1 = very unfamiliar, near 0 = very ordinary."""
    return 1 - np.asarray(p)


def main():
    """Run step 5.3 for every forecaster: s_embed of each calibration hour against the train hours.

    What: per model saves worker_b/s_embed.csv, one row per calibration hour (May-Aug 2025):
          station, timestamp, season, the 12 embedding numbers, distance, p_value, s_embed.
          Test is not read.
    How:  distance = distance of the 12-number embedding from the group shape fitted on
          train (step 5.2); p_value = conformal_pvalues against the distances of the
          train hours (2023-24) of the same season; s_embed = 1 - p_value. Copies the report to temp/.
    """
    cfg = load_config()
    w = cfg["worker_b"]
    order = group_order(cfg, w["group_by"])
    lines = []
    for model in cfg["forecasters"]["names"]:
        out = worker_dir(cfg, model)
        e_mus, e_covs, _ = load_shapes(out / w["shapes_file"])
        train, calib = load(model, "train"), load(model, "calibration")
        train_groups, calib_groups = group_of(train, w["group_by"]), group_of(calib, w["group_by"])
        d_train = own_distance(train, train_groups, e_mus, e_covs, order)
        d_calib = own_distance(calib, calib_groups, e_mus, e_covs, order)
        p = conformal_pvalues(d_train, train_groups, d_calib, calib_groups)
        emb = [c for c in calib.columns if c.startswith("emb_")]
        table = calib[["station", "timestamp", "season"] + emb].assign(distance=d_calib, p_value=p,
                                                                       s_embed=embed_score(p))
        table.to_csv(out / w["s_embed_file"], index=False)
        lines.append(f"{model}: {len(table)} calibration hours ranked against {len(train)} train hours; "
                     f"s_embed mean {table['s_embed'].mean():.3f}, share above 0.98: {(table['s_embed'] > 0.98).mean():.3f}")
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "5.3_s_embed_report.txt", text)


if __name__ == "__main__":
    main()
