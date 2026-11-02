"""Part 6 (Worker C, blocks C1-C4) for PROJECT_2: split conformal, CQR, Mondrian CQR and the adaptive level.

Ported from PROJECT steps 6.1-6.4 (same formulas). Every function takes a column key
"<target>_h<h>" (e.g. "pm25_h6"), because PROJECT_2 forecasts PM2.5 and PM10.
"""
from collections import deque

import numpy as np
import pandas as pd


def conformal_width(errors, alpha):
    """The ceil((n + 1)(1 - alpha))-th smallest value (step 6.1).

    What: returns one number; at least a (1 - alpha) share of new values is
          expected to be <= it. Returns inf if n is too small for that rank.
    How:  sort the n values and take the value at that rank (counted from 1).
          The +1 counts the new sample itself, as in the step 5.3 p-value.
    """
    errors = np.sort(np.asarray(errors))
    k = int(np.ceil((len(errors) + 1) * (1 - alpha)))
    return errors[k - 1] if k <= len(errors) else np.inf


def split_width(table, key, alpha):
    """(a) Split conformal width w from calibration errors |actual - median| (step 6.1)."""
    return conformal_width((table[f"target_{key}"] - table[f"median_{key}"]).abs(), alpha)


def cqr_scores(table, key):
    """E = max(low - actual, actual - high) per sample (step 6.2).
    What: how far the actual value is outside the model's band (positive), or
          how deep inside it is (negative).
    How:  the larger of the two one-sided misses.
    """
    actual = table[f"target_{key}"]
    return np.maximum(table[f"low_{key}"] - actual, actual - table[f"high_{key}"]).to_numpy()


def cqr_q(table, key, alpha):
    """(b) CQR correction Q = conformal_width of the CQR scores (step 6.2); range = [low - Q, high + Q]."""
    return conformal_width(cqr_scores(table, key), alpha)


def stored_scores(ref_table, ref_groups, keys):
    """The sorted CQR scores of the reference rows, per group and key (steps 6.3, 6.4).

    What: returns a dict {(group, key): sorted array of E}.
    How:  cqr_scores on the rows of each group.
    """
    ref_groups = np.asarray(ref_groups)
    return {(g, k): np.sort(cqr_scores(ref_table[ref_groups == g], k))
            for g in np.unique(ref_groups) for k in keys}


def level_score(scores, level):
    """The stored score picked by a working level, and whether the level left 0-1 (step 6.4).

    What: returns (Q, flag); flag = "below 0", "above 1" or "".
    How:  Q = the ceil((n + 1)(1 - level))-th smallest score. A level below 0
          takes the largest score, above 1 the smallest. Inside 0-1 a rank
          above n also takes the largest score (the finite stand-in for "inf").
    """
    if level <= 0:
        return scores[-1], "below 0"
    if level >= 1:
        return scores[0], "above 1"
    k = int(np.ceil((len(scores) + 1) * (1 - level)))
    return scores[min(max(k, 1), len(scores)) - 1], ""


def adaptive_ranges(table, groups, scores, key, h, alpha, gamma):
    """CQR ranges whose level moves after each known outcome, per group (step 6.4).

    What: returns (low, high, trace). trace = one row per group and hour with
          the working level and Q. gamma = 0 gives fixed (Mondrian) ranges;
          one single group gives plain CQR + the adaptive update.
    How:  hours of a group are walked in time order. At hour t:
          1. apply every update whose outcome is known by t (the outcome of hour
             s at horizon h is known at s + h): level += gamma * (alpha - miss_s),
             miss_s = average miss of the group's stations at hour s;
          2. Q = level_score(stored scores, level); range = [low - Q, high + Q];
          3. queue this hour's average miss, to be applied at t + h.
    """
    groups, times_all = np.asarray(groups), table["timestamp"].to_numpy()
    y, low, high = (table[f"{c}_{key}"].to_numpy() for c in ("target", "low", "high"))
    new_low, new_high = np.full(len(table), np.nan), np.full(len(table), np.nan)
    trace = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        order = idx[np.argsort(times_all[idx], kind="stable")]
        times = times_all[order]
        starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
        level, pending = alpha, deque()
        for s, e in zip(starts, np.r_[starts[1:], len(times)]):
            now = times[s]
            while pending and pending[0][0] <= now:
                level += gamma * (alpha - pending.popleft()[1])
            q, flag = level_score(scores[(g, key)], level)
            rows = order[s:e]
            new_low[rows], new_high[rows] = low[rows] - q, high[rows] + q
            miss = ((y[rows] < new_low[rows]) | (y[rows] > new_high[rows])).mean()
            pending.append((now + np.timedelta64(h, "h"), miss))
            trace.append({"group": g, "time": now, "level": level, "Q": q, "flag": flag})
    return new_low, new_high, pd.DataFrame(trace)
