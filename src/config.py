"""Load config.yaml and resolve its paths against the project root."""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("raw_dir", "processed_dir", "artifacts_dir", "figures_dir", "temp_dir", "era5_dir", "stations_file", "samples_dir",
             "forecasts_dir")


def load_config(path=None):
    """Return the settings as a dict; *_dir entries become absolute Paths.

    The file is taken from `path`, else the AQI_CONFIG environment variable,
    else config.yaml in the project root.
    """
    path = Path(path or os.environ.get("AQI_CONFIG") or PROJECT_ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = path.resolve().parent
    for key in PATH_KEYS:
        cfg[key] = (root / cfg[key]).resolve()
    return apply_run(cfg)


def apply_run(cfg):
    """Point the Part 3-7 outputs of one run to their own folders.

    What: with AQI_IMPUTATION=<method> (and optionally AQI_INPUT_HOURS=<h>) set,
          the run "<method>" or "<method>_w<h>" writes features and model table to
          processed_dir/<runs>/<run>/, samples and forecasters to artifacts_dir/<runs>/<run>/,
          figures to figures_dir/<runs>/<run>/ and temp files to temp_dir/<runs>/<run>/.
          Steps 2.x keep reading cpcb_clean from processed_dir. Without the
          variables nothing changes (config.yaml method, top-level folders).
    How:  file names that live in processed_dir become absolute paths, which
          `processed_dir / name` keeps as they are.
    """
    method, hours = os.environ.get("AQI_IMPUTATION"), os.environ.get("AQI_INPUT_HOURS")
    if not method:
        return cfg
    imp = cfg["imputation"]
    if method not in imp["methods"]:
        raise SystemExit(f"AQI_IMPUTATION={method} is not one of {imp['methods']}")
    imp["method"] = method
    run = method
    if hours:
        cfg["samples"]["input_hours"] = int(hours)
        run = f"{method}_w{hours}"
    sub = Path(imp["runs_folder"]) / run
    data_dir = cfg["processed_dir"] / sub
    data_dir.mkdir(parents=True, exist_ok=True)
    for key in ("features_file", "model_table_file"):
        cfg[key] = str(data_dir / cfg[key])
    cfg["samples_dir"] = cfg["artifacts_dir"] / sub / "samples"
    cfg["forecasts_dir"] = cfg["artifacts_dir"] / sub / "forecasters"
    cfg["figures_dir"] = cfg["figures_dir"] / sub
    cfg["temp_dir"] = cfg["temp_dir"] / sub
    cfg["run_name"] = run
    return cfg
