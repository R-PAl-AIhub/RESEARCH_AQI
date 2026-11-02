"""Save a step's printed check and intermediate tables into temp/ for inspection."""


def _temp_file(cfg, name):
    """Full path of a file inside temp_dir.

    What: returns temp_dir / name, creating temp_dir if it does not exist yet.
    How:  temp_dir comes from config.yaml (already an absolute Path); mkdir with
          exist_ok=True does nothing if the folder is already there.
    """
    folder = cfg["temp_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def save_text(cfg, name, text):
    """Write the printed check of a step to temp/<name>.

    What: saves `text` (what was shown on screen) as a .txt file.
    How:  opens the file in write mode with UTF-8 (so µ and other symbols survive)
          and replaces any older version from a previous run.
    """
    path = _temp_file(cfg, name)
    path.write_text(text, encoding="utf-8")
    return path


def save_table(cfg, name, df):
    """Write an intermediate table to temp/<name> as CSV (opens in Excel).

    What: saves a pandas table, keeping its index (e.g. station, year).
    How:  DataFrame.to_csv; an older file with the same name is replaced.
    """
    path = _temp_file(cfg, name)
    df.to_csv(path)
    return path
