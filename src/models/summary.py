"""Part 4: print model.summary() (layers, output shapes, weight counts) of every forecaster."""
import json

from src.config import load_config
from src.models.forecasters import build


def summaries(cfg, n_hours, n_features, n_targets, n_horizons, n_stations):
    """The model.summary() text of every forecaster, point and quantile.

    What: returns one text block per model and kind: layer name and type,
          output shape, number of weights, and the total.
    How:  builds each model fresh from config.yaml (same code and sizes as
          training, so the layers are identical; no training needed) and
          collects summary() lines through print_fn.
    """
    blocks = []
    for name in cfg["forecasters"]["names"]:
        for alpha in (None, cfg["alpha"]):
            model, _ = build(name, n_hours, n_features, n_targets, n_horizons, cfg["d"],
                             cfg["forecasters"], cfg["model"], alpha, n_stations)
            lines = []
            model.summary(print_fn=lambda line, **_: lines.append(line), line_length=110)
            blocks.append("\n".join(lines))
    return blocks


def main():
    """Write every summary to figures_dir/forecast_summaries.txt and print it.

    What: input (samples, 48 hours, 53 columns); outputs (2 targets x 5 horizons),
          x 3 lines (low, median, high) for the quantile models.
    How:  sizes come from the step 3.3 files (feature_names.json, target_layout.json).
    """
    cfg = load_config()
    samples_dir = cfg["samples_dir"]
    n_features = len(json.loads((samples_dir / "feature_names.json").read_text(encoding="utf-8")))
    layout = json.loads((samples_dir / "target_layout.json").read_text(encoding="utf-8"))
    text = "\n\n".join(summaries(cfg, cfg["samples"]["input_hours"], n_features,
                                 len(layout["targets"]), len(layout["horizons"]),
                                 len(json.loads((samples_dir / "stations.json").read_text(encoding="utf-8")))))
    out = cfg["figures_dir"] / "forecast_summaries.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
