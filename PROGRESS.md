2.1 | 2026-10-13 | read_one() copied; metadata fix (value in col C); saves temp/2.1_one_file.csv only | Ballygunge 2025: (7388, 21), 2025-01-01 to 12-31 23:00, PM2.5 missing 140
2.2 | 2026-10-13 | read_all() copied -> data/processed/cpcb_hourly.csv (CSV only) | 21 files, 179559 raw rows, 0 dups, 26304 h/station, 184128 grid rows
2.3 | 2026-10-13 | availability.py + chart availability_targets.png; config targets = PM2.5, PM10 | table same as PROJECT; lowest Fort William 2025: PM2.5 6840 h, PM10 6759 h
config | 2026-10-13 | flags.range set to official limits: EPA 2016 (gases, CO added), NC DEQ/EPA (PM -10), WMO 2004 (RH, BP 500-1100, AT -80..60, WD, SR 0-1600) | not run yet
config | 2026-10-13 | NO -1.23, NOX -1 ppb, NH3 -0.70 = -LDL (EPA 2011 -MDL rule + CPCB CAAQM spec LDL 1 ppb) | CPCB spec says BP is mmHg: BP range open
2.4-2.5 | 2026-10-13 | range_clean.py -> cpcb_clean.csv + range_check.png; drop XYLENE/MP-Xylene/BENZENE/TOLUENE | 380 readings emptied (all RBU 2023-24: NH3 121, NO 97, SR 66, NOX 57, RH 27, NO2 12)
2.7 | 2026-10-13 | merge_era5.py -> merged.csv (184128, 21); ERA5 u10, v10 only + net speed era5_ws | 7 stations -> 3 grid points; 5 rows/station without ERA5 (first hours of 2023)
3.1 | 2026-10-13 | plan.py, hours with PM2.5 AND PM10 | 174425 h; k=4 -> 91 days/cluster, d=8 PASS; Red 0.02 PASS (ON 61>=49); splits kept from PROJECT
3.2 | 2026-10-13 | build.py -> features.csv (184128, 40); target_pm25 + target_pm10 | 34 features (era5 = u10, v10, ws only); longest fill 6 h; 5/5 tests pass
3.3 | 2026-10-13 | samples.py -> artifacts/samples; y (N, 2 targets, 4 horizons) | X (N,48,37); train/calib/test 52597/55108/51699; train mean 0 sd 1
config | 2026-10-13 | ranges CPCB-first (analyser + met sensor ranges, RF added); BP hPa->mmHg (CPCB unit); reran 2.5-3.3 | 439 emptied (+SO2 56, SR 2, RF 1); samples unchanged; 5/5 tests
2.4 | 2026-10-13 | flags.py: stuck-sensor check only (PM 12+ h same value), values kept; spike rule removed; flag_check.png | 104 runs: PM2.5 1930 h, PM10 539 h (most: Ballygunge, Rabindra Sarobar, Fort William)
rerun | 2026-10-13 | old plots/logs deleted; full rerun 2.1-3.3 fresh | merged.csv (184128, 21) = final pre-Part-3 table; samples 52597/55108/51699; 5/5 tests
2.5 | 2026-10-13 | NO value changed: range (439) + stuck (PM2.5 1930 h, PM10 539 h) only counted; BP kept in hPa (mmHg only inside check) | merged.csv CPCB values == source: True
2.5 | 2026-10-13 | rows with PM2.5 or PM10 stuck 12+ h dropped (list in stuck_rows_dropped.csv); other values unchanged | 2129 rows dropped; merged.csv (181999, 21); Part 3 not rerun
2.5 | 2026-10-13 | drop reversed; 0/1 columns stuck_pm25, stuck_pm10 added (12+ h same value) | stuck_pm25=1 on 1930 h, stuck_pm10=1 on 539 h; merged.csv (184128, 23); all values kept
2.5/2.7 | 2026-10-13 | RF column removed (config clean.drop_columns, RF range rule deleted) | merged.csv (184128, 22); 438 out-of-range counted; values kept
2.5/2.7 | 2026-10-13 | CO column removed (2023-24 ~80% zeros from whole-number rounding; 2025 decimals) | merged.csv (184128, 21); values kept
3.2 | 2026-10-13 | build.py -> features.csv (184128, 41); + PM10_mean24h; season labels winter/summer_premonsoon/sw_monsoon/post_monsoon (config) | 35 features; fill <=6 h past only; targets untouched; 2/2 feature tests pass
3.1 | 2026-10-13 | plan.py with config seasons (winter/summer_premonsoon/sw_monsoon/post_monsoon), PM2.5 AND PM10 hours | 174425 h; train/calib/test days 364/366/365; k=4 91 days -> d=8 PASS; Red 0.02 PASS (post_monsoon 61>=49); GATE approved
3.1 | 2026-10-13 | k removed: 4 natural seasons = clusters (plan.py, config) | smallest train season post_monsoon 61 d -> d=8 FAIL (max d 6); levels 0.02/0.10/0.05 PASS; GATE: d open
3.1 | 2026-10-13 | d set to 6 (config) | d=6 needs 60 train days, post_monsoon has 61 -> PASS; step 3.1 GATE approved and closed
3.3 | 2026-10-13 | samples.py: miss_PM/miss_GAS/miss_METEOROLOGICAL (CO, RF out); model_table.csv (184128, 46) to view | X (N,48,38); train/calib/test 52754/55140/51771; 5/5 tests
3.3 | 2026-10-13 | model_table.csv = station, time + the 38 model input columns, empty -> 0 (what X holds) | (184128, 40); 0 empty cells; X row == CSV row: True
4.1-4.2 | 2026-10-13 | 5 forecasters (GRU, LSTM, CNN, CNN+LSTM, simplified TFT), point + quantile; one d=6 embedding serves 2 targets x 4 horizons | heads read embedding only: True x10
4.x | 2026-10-13 | compare.py: calib RMSE vs no change, skill, loss, coverage, bands, per station/year | all beat no change at +6/+12 h, all lose at +1 h; GRU best PM2.5 +12/+24 h
5.2-5.6b | 2026-10-13 | Worker B with seasons as clusters (no 5.1 k-means, no 5.4 Simes), all 5 models; raw net per season | max cond 946 (<1000); check-week Red 1.5-4.1%
7.1 | 2026-10-13 | decide.py on test 2025: base = adaptive Mondrian by season, Yellow 0.5 x level, flags shared by both targets | Red 1.9-2.7% of test samples
7.2 | 2026-10-13 | stratified.py: 6 methods ((e)=(d)), normal vs top-5% hours, per season and station, 1000 day-block bootstraps | normal ~91-92%; extreme 60-85% (ours), split 12-39%
2.3-2.5 | 2026-10-15 | drop only XYLENE, MP-Xylene; keep BENZENE, TOLUENE, RF, CO; reran 2.3-2.5 | cpcb_clean.csv (184128, 22); 438 out of range; values kept; 2.7+ not rerun
2.5 | 2026-10-15 | row_present dropped from cpcb_clean (config clean.drop_columns); still in cpcb_hourly for 2.3 | cpcb_clean.csv (184128, 21); values kept: True
2.7 | 2026-10-16 | ERA5 wind step removed: 3.1 plan.py and 3.2 build.py read cpcb_clean.csv; merged.csv no longer used | not rerun yet (3.1+ pending)
2.1-2.5 | 2026-10-16 | full rerun 2.1-2.5 (WSL venv) with current config | 2.1 (7388, 21); 2.2 184128 grid rows; stuck 1930/539 h; cpcb_clean (184128, 21); values kept: True
3.1 | 2026-10-16 | option C: train 2023-24, val Jan-Apr, calib May-Aug, test Sep-Dec 2025; group_by all; d = 12 (max per season) | hours 118027/19515/17196/19687; levels PASS
3.2 | 2026-10-16 | build.py reads cpcb_clean (no ERA5); + CO, BENZENE, TOLUENE, RF | features.csv (184128, 41); 36 features; longest fill 6 h; targets unfilled
3.2 | 2026-10-16 | rolling mean + std of past 12 h and 24 h for PM2.5, PM10 (config rolling_hours/rolling_stats) | features.csv (184128, 47); 42 features; 2/2 leak tests pass
3.2-3.3 | 2026-10-16 | rolling 6/12/24 h; ERA5 CSVs to Recycle Bin; 4-part split; CO/BENZENE/TOLUENE in miss_GAS, RF in miss_MET | 46 features; X 48x49; train/val/calib/test 108672/17748/15557/17716; 5/5 tests
2.1-3.3 | 2026-10-16 | + rolling 3 h (mean, std); full rerun 2.1-3.3 (WSL) | 50 features; X 48x53; samples 108677/17746/15540/17730; values kept; 5/5 tests
4.x | 2026-10-16 | 5 models x point/quantile, val = Jan-Apr 2025, d = 12; compare on calib May-Aug 2025; charts per split | all beat no change at +6/+12 h; GRU lowest; 10/10 embed-only
5.2-5.6b | 2026-10-16 | Worker B one group (group_by all); seasons only labels | cond <= 191 (raw 467); check p<0.10 9-11%; check Red 1.5-2.9%
7.1-7.2 | 2026-10-16 | test Sep-Dec 2025 (v2; v1 already saw 2025) | test Red 2-16% (calib calm, test smoggy); PM2.5 normal 92-94% ours; smoggy 58-69%; 15/15 tests
3.1 | 2026-10-16 | option C approved: train 2023-24, val Jan-Apr, calib May-Aug, test Sep-Dec 2025; one group; d = 73 (max) | hours 118027/19515/17196/19687; levels PASS
+3 h | 2026-10-17 | horizons 1/3/6/12/24; beta +3 h = 0.4; stratified grid fits 5; rerun 3.3-7.2 (WSL) | samples 107440/17549/15373/17562; test Red 2-18%; 15/15 tests
5.3 | 2026-10-17 | p-value check: calibration May-Aug 2025 ranked against train 2023-24 (no even/odd weeks); pvalue.py + chart text; 5.6/5.6b unchanged | share p<0.10: 0.9-9.5%; 15/15 tests
5.3-7.2 | 2026-10-17 | s_embed = rank of calibration embedding distance among train 2023-24 distances; checks removed; s_embed.csv per model; reran 5.6-7.2 | test Red 2-9%; 15/15 tests
5.6b-7.2 | 2026-10-19 | label_h<h> now straight from smoothed_h<h> (cuts 0.80 / 0.98); final_h<h> removed; reran 5.6b-7.2 | test Red 1-3%; calib Green 73-83%; 15/15 tests
6.3 | 2026-10-19 | old 6.3 box removed from flowchart/index.html (one group = same as 6.2); adaptive update renamed 6.4 -> 6.3 | flowchart only, no code or data change
7.1 | 2026-10-19 | flowchart 7.1 box: base_ = 90% range, yellow_ = 95% range, lower_/upper_ = range used (base if Green, yellow if Yellow, empty if Red) | flowchart only
7.2 | 2026-10-19 | methods (d) groups and both (f) ours removed; 3 methods left: split, CQR, CQR + adaptive; reran 7.2 | CQR+adaptive PM2.5 normal 91-92%, extreme 49-83%; 15 tests pass
4.x | 2026-10-25 | "no change" baseline removed: compare.py, train.py, config; skill chart dropped; point report.txt RMSE rewritten (no retrain); all 15373 calib samples scored | 15/15 tests
rerun | 2026-10-25 | no-fill trial reverted (6 h fill back); full rerun 3.2-7.2, all 10 models retrained, no "no change" | samples 107440/17549/15373/17562; test Red 1-3%; 15/15 tests
2.3 | 2026-10-25 | availability_all.png + availability_all_hours.csv: hours with a value for all 17 kept columns (not XYLENE, MP-Xylene); view only | reran 2.3; nothing later changes
5.6b | 2026-10-25 | label-share charts: y-axis now "% of calibration week samples" (was check-week); charts redrawn | plots only
5.6b | 2026-10-25 | label-share chart titles: "check-week" -> "calibration week"; charts redrawn | plots only
2.3 | 2026-10-26 | missing_table.csv/.png (+ per station): missing hours, count and %, per year x 17 variables (paper style) | 2025 worst: BENZENE/TOLUENE 9.1%
3.2-7.2 | 2026-10-26 | 6 imputation runs (deletion, median, ffill, bfill, knn, mice) x 5 models, full TFT, 48 h, own runs/ folders; picker in index.html | 25/25 tests
compare | 2026-10-26 | same 4659 calib hours: median 8.39 ~ ffill 8.42 (tie within retrain noise; bfill 8.40 uses future), mice 8.52, knn 8.55, deletion worst 9.01
charts | 2026-10-27 | src/plot_style.save_figure for every chart: no grid, one legend outside right-most panel, sub/superscripts (PM2.5, NO2, s_embed...); all PNGs redrawn
charts | 2026-10-27 | "uses future" labels removed from comparison chart and flowchart picker (bfill still never picked); old run got stations.json
3.3 | 2026-10-27 | deletion_hours.csv/.png (deletion run): hours with an empty input cell per station x year + total; linked in 3.3 box | total 40117 of 184128 h (21.8%); Fort William 2025 46.4%
3.3 | 2026-10-27 | flowchart 3.3 bullet per method: filled methods skip a window only when a PM answer is missing (after MICE only 168 h on 1 Jan 2023 have empty lags)
3.3 | 2026-10-27 | miss_PM/GAS/METEOROLOGICAL flags removed: 48 h x 50 input columns; all 6 runs retrained (3.2 -> 7.2), 25 tests pass
5.5 | 2026-10-27 | beta smoothing step removed (smooth.py to Recycle Bin): label = fused score straight, same for every horizon
5.2 | 2026-10-27 | Worker B by 4 seasons, plain mean+covariance (MCD+LW removed), condition limit 100000; s_embed and s_raw ranked vs same-season train
6.3 | 2026-10-27 | Worker C one group; gamma search on calib alternating weeks (closest to 90%, tie 0.5 pt -> narrower): gamma 0.001 for all runs
7.2 | 2026-10-27 | imputation_compare.png 2x2: calib RMSE PM2.5/PM10 + test coverage smoggiest 5%; best method ffill (common RMSE 8.397)
flow | 2026-10-27 | flowchart/index_2.html built (no MCD/LW, no beta, no flags; gamma grid + shape_compare links); index.html left unchanged
5.2 | 2026-10-27 | shape_compare.png: plain vs MCD+LW; robust flags ~72% Yellow on calib, test Red/Green error 0.70-0.84 (<1) vs plain 1.18-1.48 -> plain kept
6.3 | 2026-10-27 | gamma_horizon_check.png + gamma_station_check.png/.csv: every horizon wants gamma 0-0.002; stations at 0.001 span 85-94% (Ballygunge PM10 74-83%)
arch | 2026-10-27 | AQI_Reliability_Layer_Architecture_v3.md in PROJECT_2 (v2 untouched): B3 plain, B4 own season, B5-B6 vs train, B7 removed, fused s, C3 one group, C4 gamma 0.001
deck | 2026-10-27 | presentation/Reliable_PM_Forecasts.pptx: 15-slide Canva deck of the flowchart (template copy), exported as PowerPoint
deck | 2026-10-27 | Reliable_PM_Forecasts_with_charts.pptx: 37 slides = 15 Canva + 22 chart slides in flowchart order, captions + speaker notes
deck | 2026-10-27 | chart slides now have clickable links to the chart PNG and its CSV files (50 links, all checked to open)
flow | 2026-11-01 | index_2.html: box 5.2 lost MCD + condition-number lines (box shortened); titles renumbered 1, 2.1-2.2, 3.x, 4-4.2, 5.1-5.4, 6.x, 7.x
deck | 2026-11-02 | both pptx renumbered to match index_2 (2.2, 4.2, 5.1, 5.3, 5.4; slide 6 steps 2.1-2.2)
metrics | 2026-11-02 | src/eval/metrics_table.py: RMSE/MAE/R² of published forecast, 4 splits x 6 runs x 5 models -> metrics_table.csv/summary + table PNGs
r2 | 2026-11-02 | forecast_r2.png per run (R² by horizon, calibration) replaces the RMSE chart; model colours = colourblind-checked palette
shap | 2026-11-02 | src/eval/shap_explain.py (shap 0.54, GRU ffill, 500 calib windows): importance, beeswarm x2, correlation; checks 0.03 / 2.15 µg/m³
charts | 2026-11-02 | src/eval/modern_charts.py: scatter, error boxes, imputation dots, coverage dumbbells, station dots, flag bars, smoggy days, availability
clean | 2026-11-02 | MCD compare removed (shape_compare png/csv/py to Recycle Bin, config block, 5.1 chips, deck slide); index.html = index_2 (old kept as index_old.html)
coverage | 2026-11-02 | src/eval/coverage_table.py: published 90% range (test) + model band (2023-25) per station, run, model; CSV + 4 PNGs; check = step 7.2
flow | 2026-11-02 | index.html + index_2: R² chip replaces RMSE; 23 new chips (metrics, SHAP, modern charts, coverage A/B) in boxes 2.2, 3.2, 4, 7.1, 7.2; side label 47 raw columns
shap | 2026-11-02 | shap_share.png: share of total SHAP per input group x horizon + all 50 inputs ranked; linked in flowchart box 4
shap | 2026-11-02 | shap_share.png: share of total SHAP per input group x horizon + all 50 inputs ranked; linked in flowchart box 4
eval | 2026-11-02 | src/eval/all_runs_charts.py: R², RMSE (4 splits) and test coverage heatmaps, every run x model x horizon (6 PNGs)
charts | 2026-11-02 | modern_charts now loops every run x model: 134 charts in figures/runs/<run>/(<model>/)
shap | 2026-11-02 | shap_explain loops every run x model (TFT: direct expected gradients, checked vs shap lib); original input names; full run pending
flow | 2026-11-02 | index.html/index_2: result bullets in 3.2, 4, 7.1, 7.2; chips now follow run/model pickers; all-runs heatmaps linked
github | 2026-11-02 | pushed website set to github.com/R-PAl-AIhub/RESEARCH_AQI (1,719 files, 273 MB): figures, RECENT_PLOTS (211) + gallery, landing index.html
github | 2026-11-02 | SHAP for all 30 run x model pairs added; RECENT_PLOTS + gallery refreshed and pushed
deck | 2026-11-02 | deck 36 -> 50 slides: RMSE slide replaced by R²; metrics table, R² all runs, scatter, boxes, 4 SHAP, flags, coverage all/A/B, availability, imputation dots
