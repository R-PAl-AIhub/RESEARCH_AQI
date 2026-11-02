"""Step 5.2 (Worker B, blocks B3-B4): shape (mean + covariance) of each season and the distance to it.

PROJECT_2: the groups of Worker B are the 4 natural seasons (worker_b.group_by: season; no k-means,
step 5.1 skipped), so a sample's distance is measured to its own season's shape. The shape is the plain
mean and covariance of the train hours of that season (MCD and Ledoit-Wolf removed 2026-10-27).
Worker C keeps one group (worker_c.group_by: all).
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import load
from src.temp_out import save_table, save_text

PERCENTILES = (50, 90, 99)


def season_order(cfg):
    """The 4 season labels in calendar order: winter, summer_premonsoon, sw_monsoon, post_monsoon.

    What: returns a list of the distinct values of config features.seasons.
    How:  dict.fromkeys keeps the first-seen order (months 12, 1, 2, then 3 ...).
    """
    return list(dict.fromkeys(cfg["features"]["seasons"].values()))


def group_order(cfg, group_by):
    """The group labels of one worker: ["all"] (group_by "all") or the 4 seasons (group_by "season").

    What: returns the list of group labels in a fixed order.
    How:  Worker B passes worker_b.group_by (season: its shapes and ranks come from train,
          which has every season); Worker C passes worker_c.group_by (all: the May-Aug
          calibration has no winter or post-monsoon hours).
    """
    return ["all"] if group_by == "all" else season_order(cfg)


def group_of(table, group_by):
    """The group label of every row: "all" for every row, or the row's season.

    What: returns an array with one label per row of `table` (needs a season column).
    How:  group_by "all" -> "all" everywhere; group_by "season" -> the season column.
    """
    if group_by == "all":
        return np.full(len(table), "all", dtype=object)
    return table["season"].to_numpy()


def embedding_matrix(table):
    """The emb_1 .. emb_d columns of an exported forecaster table, as an array (samples, d)."""
    return table[[c for c in table.columns if c.startswith("emb_")]].to_numpy()


def worker_dir(cfg, model):
    """forecasts_dir/<model>/worker_b/, created if missing (every Part 5 output of that model)."""
    folder = cfg["forecasts_dir"] / model / "worker_b"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def cluster_shape(points, max_condition):
    """Centre and covariance of one group: the plain mean and covariance of all its points.

    What: returns a dict with mu (centre), cov (covariance), the condition number
          and the number of points.
    How:  every point counts (no trimming, no shrinkage). Condition number = largest /
          smallest spread direction; above `max_condition` the inverse is unreliable,
          so the run stops.
    """
    shape = {"mu": points.mean(axis=0), "cov": np.cov(points, rowvar=False), "n_points": len(points)}
    shape["condition"] = np.linalg.cond(shape["cov"])
    if shape["condition"] > max_condition:
        raise SystemExit(f"condition number {shape['condition']:.0f} above {max_condition}: "
                         f"stop and decide (smaller d or more data).")
    return shape


def fit_shapes(features, seasons, order, w_cfg):
    """cluster_shape for every group of any feature table (step 5.6 reuses this).

    What: returns (mus (groups, n), covs (groups, n, n), report table one row per group).
    How:  takes the rows of `features` whose group label is that group, in `order`.
    """
    shapes = [cluster_shape(features[seasons == s], w_cfg["max_condition"]) for s in order]
    report = pd.DataFrame([{k: s[k] for k in ("n_points", "condition")}
                           for s in shapes], index=pd.Index(order, name="group")).astype(float).round(3)
    return np.stack([s["mu"] for s in shapes]), np.stack([s["cov"] for s in shapes]), report


def distance(h, mus, covs, cluster=None):
    """Mahalanobis distance of each point to a cluster.

    What: returns (d, cluster index). cluster = None -> the nearest cluster
          (min over k); cluster = an index array -> the distance to that row's
          own cluster (PROJECT_2: its season).
          d = sqrt((h - mu_k)^T Sigma_k^-1 (h - mu_k)).
    How:  a step along a direction where the cluster is thin counts more than
          the same step along a direction where it is wide.
    """
    h = np.atleast_2d(h)
    all_d = np.stack([np.sqrt(np.einsum("ij,jk,ik->i", h - mu, np.linalg.inv(cov), h - mu))
                      for mu, cov in zip(mus, covs)], axis=1)
    if cluster is None:
        return all_d.min(axis=1), all_d.argmin(axis=1)
    cluster = np.asarray(cluster)
    return all_d[np.arange(len(h)), cluster], cluster


def season_index(seasons, order):
    """Season labels -> their position in `order` (0 .. 3), the cluster id."""
    lookup = {s: i for i, s in enumerate(order)}
    return np.array([lookup[s] for s in seasons])


def own_distance(table_or_points, seasons, mus, covs, order):
    """Distance of each sample to its own season's shape (embedding table or a point array)."""
    points = embedding_matrix(table_or_points) if isinstance(table_or_points, pd.DataFrame) else table_or_points
    return distance(points, mus, covs, season_index(seasons, order))[0]


def load_shapes(path):
    """The saved centres and covariances: returns (mus, covs, season order)."""
    saved = np.load(path)
    return saved["mu"], saved["cov"], list(saved["seasons"])


def toy_check():
    """Check 1: the distance function on a covariance given directly.

    What: two features, centre 0, spread 1, correlation -0.7; returns the
          distances of (-2, +2) and (+2, +2), which must be 2.17 and 5.16.
    How:  (-2, +2) goes along the cluster's long diagonal (the features move in
          opposite directions), so it is close; (+2, +2) goes across it, so it is far.
    """
    cov = np.array([[[1.0, -0.7], [-0.7, 1.0]]])
    d, _ = distance(np.array([[-2.0, 2.0], [2.0, 2.0]]), np.zeros((1, 2)), cov)
    return d


def main():
    """Run step 5.2 for every forecaster: fit each season's shape on TRAIN embeddings, save, check.

    What: per model saves worker_b/embed_shapes.npz and worker_b/distances.csv
          (train + calibration: station, timestamp, season, split, distance);
          prints per-season condition numbers, the toy check, and
          the 50th / 90th / 99th percentile of the distance per split and season.
          The test split is not read. Copies to temp/.
    How:  settings from config.yaml (worker_b section and seed).
    """
    cfg = load_config()
    w_cfg = cfg["worker_b"]
    group_by = w_cfg["group_by"]
    order = group_order(cfg, group_by)
    toy = toy_check()
    lines = [f"groups = {order} (worker_b.group_by: {group_by}); plain mean + covariance of train, "
             f"condition limit {w_cfg['max_condition']}",
             f"check 1 (toy): (-2,+2) -> {toy[0]:.2f} (expect 2.17)   (+2,+2) -> {toy[1]:.2f} (expect 5.16)   "
             f"{'PASS' if np.allclose(toy.round(2), [2.17, 5.16]) else 'FAIL'}"]
    for model in cfg["forecasters"]["names"]:
        out = worker_dir(cfg, model)
        train = load(model, "train")
        mus, covs, report = fit_shapes(embedding_matrix(train), group_of(train, group_by), order, w_cfg)
        np.savez(out / w_cfg["shapes_file"], mu=mus, cov=covs, seasons=np.array(order),
                 condition=report["condition"].to_numpy())

        parts = []
        for split, table in (("train", train), ("validation", load(model, "validation")),
                             ("calibration", load(model, "calibration"))):
            groups = group_of(table, group_by)
            d = own_distance(table, groups, mus, covs, order)
            parts.append(table[["station", "timestamp", "season"]].assign(group=groups, split=split, distance=d))
        dist = pd.concat(parts, ignore_index=True)
        dist.to_csv(out / "distances.csv", index=False)
        spread = dist.groupby(["split", "season"])["distance"].quantile([p / 100 for p in PERCENTILES]).unstack()
        spread.columns = [f"p{p}" for p in PERCENTILES]
        spread = spread.round(2)
        lines += [f"\n===== {model} =====", report.to_string(),
                  "distance to the own season's shape, percentiles per split and season:",
                  spread.to_string()]
        save_table(cfg, f"5.2_{model}_season_shapes.csv", report)
        save_table(cfg, f"5.2_{model}_distance_percentiles.csv", spread)
    text = "\n".join(lines)
    print(text)
    save_text(cfg, "5.2_distance_report.txt", text)


if __name__ == "__main__":
    main()
