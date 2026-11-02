"""Alternating-week split of the calibration year (every later two-part check uses this)."""


def alternating_weeks(timestamps):
    """Label each time "reference" (odd calendar week) or "check" (even calendar week).

    What: returns a pandas Series of "reference" / "check", one per timestamp.
    How:  ISO calendar week number of each time; odd -> reference, even -> check.
          Alternating weeks give both parts every season, unlike two halves of
          the year, which would put winter in one part and monsoon in the other.
    """
    week = timestamps.dt.isocalendar().week.astype(int)
    return week.mod(2).map({1: "reference", 0: "check"})
