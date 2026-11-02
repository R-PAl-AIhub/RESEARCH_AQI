"""Part 4: forecast errors per horizon, in µg/m³."""
import numpy as np
import pandas as pd


def mse(y_true, y_pred):
    """Mean squared error per column (one column = one horizon).

    What: returns one number per horizon, in (µg/m³)²; big misses count extra
          because errors are squared before averaging.
    How:  mean((prediction - truth)²) down each column; missing values skipped.
    """
    return np.nanmean((np.asarray(y_pred) - np.asarray(y_true)) ** 2, axis=0)


def rmse(y_true, y_pred):
    """Root mean squared error per column (one column = one horizon).

    What: returns one number per horizon, back in µg/m³.
    How:  square root of mse.
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean absolute error per column (one column = one horizon).

    What: returns one number per horizon = the average size of a miss.
    How:  mean(|prediction - truth|) down each column; missing values skipped.
    """
    return np.nanmean(np.abs(np.asarray(y_pred) - np.asarray(y_true)), axis=0)


def error_rows(y_true, predictions, targets, horizons):
    """RMSE, MSE and MAE of several forecasts, as a long table.

    What: one row per forecast, target and horizon with columns model, target,
          horizon, rmse, mse, mae. y_true and every prediction are arrays of
          shape (samples, targets, horizons) in µg/m³.
    How:  for each target, calls the three metrics on that target's slice;
          rounded to 2 decimals.
    """
    rows = []
    for name, pred in predictions.items():
        for t, target in enumerate(targets):
            values = {m: f(y_true[:, t], pred[:, t]) for m, f in (("rmse", rmse), ("mse", mse), ("mae", mae))}
            rows += [{"model": name, "target": target, "horizon": h,
                      **{m: round(float(v[i]), 2) for m, v in values.items()}} for i, h in enumerate(horizons)]
    return pd.DataFrame(rows)
