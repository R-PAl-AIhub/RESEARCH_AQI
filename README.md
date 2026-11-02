# Kolkata AQI reliability layer (PROJECT_2)

PM2.5 and PM10 forecasts for 7 Kolkata CPCB stations (2023–2025) with a reliability layer:
5 forecasters → Worker B trust flag (Green / Yellow / Red) → Worker C conformal 90% range → decision → evaluation.

**Website:** https://r-pal-aihub.github.io/RESEARCH_AQI/ (flowchart: `flowchart/index.html`, new charts: `RECENT_PLOTS/index.html`).

- `src/` code, `tests/` tests, `config.yaml` all settings
- `figures/` every chart and table (`figures/runs/<imputation run>/<model>/`)
- `RECENT_PLOTS/` copies of the November 2026 charts
- `temp/` printed reports of every step, `Kolkata/` raw CPCB files (read only)
- Not included (too large for GitHub): `artifacts/` (trained models, samples) and `data/processed/`
