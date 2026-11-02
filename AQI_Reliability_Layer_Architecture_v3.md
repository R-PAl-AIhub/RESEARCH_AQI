        # AQI Reliability Layer — Architecture v3 (PROJECT_2, Kolkata)
### (Worker B = Regime/OOD · Worker C = Conformal)
*Reflects everything refined in the step-by-step teaching, Parts 1–4. Supersedes the v1 flowchart on the OOD side (v1 used raw features + χ² p-values — both replaced here).*
*v3 (2026-10-27) = v2 changed to match the PROJECT_2 pipeline as run: B3 plain covariance, B4 own season, B5–B6 ranked against same-season train, B7 removed, fused score, C3 one group, C4 γ = 0.001. Each change is marked **[v3]**.*

---

## 0. The one sentence the whole design hangs on

> **Conformal prediction's honest ± range is only valid IF today is "exchangeable" with the calibration days. Worker B is the statistical test of that IF. Worker C produces the range once the IF is judged to hold — per regime.**

So the data flows **B → C**: B decides *which bag today belongs to*, C produces *the honest width for that bag*.

---

## 1. Full pipeline (upgraded)

```
        ┌──────────────────────────────────────────────────────────────┐
        │  STAGE 1–2 : DATA + PREPROCESSING   (shared, unchanged)       │
        │  CPCB ground · ERA5 · satellite AOD → align to stations →     │
        │  clean · impute · lag features · cyclic time encodings →      │
        │  split : TRAIN / CALIBRATION / TEST  (time-ordered, no leak)  │
        └───────────────────────────────┬──────────────────────────────┘
                                         ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  STAGE 3 : PREDICTION ENGINE   (physics + attention + graph)  │
        │  advection–diffusion → spatial attn → temporal attn + GRU     │
        │                                                               │
        │   ── produces TWO things we consume downstream ──             │
        │   (1) multi-horizon forecast  ŷ  at +1/+6/+12/+24 h           │
        │   (2) internal embedding  h_t  = the model's "notepad"        │
        │       (penultimate layer, per station, per horizon)          │
        └───────────────┬───────────────────────────┬──────────────────┘
                        │ h_t (notepad)             │ ŷ (forecast)
                        ▼                            ▼
   ╔════════════════════════════════════╗   ╔════════════════════════════════════╗
   ║  WORKER B : REGIME / OOD           ║   ║  WORKER C : CONFORMAL              ║
   ║  "which bag is today?"             ║   ║  "honest ± width for this bag"     ║
   ║  (DETAILED BELOW — Parts 1–4)      ║   ║  (DETAILED in Parts 5–7 — stub)    ║
   ╠════════════════════════════════════╣   ╠════════════════════════════════════╣
   ║ B1 distance in NOTEPAD space (P2)  ║   ║ C1 nonconformity score on          ║
   ║    + raw-space distance (baseline) ║   ║    calibration residuals           ║
   ║ B2 per-season centres μ_k,Σ_k (P3) ║   ║ C2 CQR-style width (heteroskedastic)║
   ║ B3 plain mean + covariance  [v3]   ║   ║ C3 ONE group for all hours  [v3]   ║
   ║ B4 d = Mahalanobis to own season   ║   ║    (calibration has 2 seasons only) ║
   ║ B5 empirical-rank score, NOT χ²    ║   ║ C4 (drift) adaptive update,         ║
   ║    (P1 fix, P4)                     ║   ║    γ = 0.001  [v3]                  ║
   ║ B6 +1 p-value vs same-season train ║   ║ output: interval width  w           ║
   ║ B7 (removed: no smoothing)  [v3]   ║   ╚════════════════════════════════════╝
   ║ B8 seasonal calibration            ║                    ▲
   ║                                    ║   label k (picks   │
   ║ output: score s , label k ∈{G,Y,R} ╠═══ the range) ═════╝
   ╚════════════════┬═══════════════════╝
                    │  s , k                       │  w
                    └───────────────┬──────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  STAGE 5 : DECISION + OUTPUT                                   │
        │   GREEN  (s ≤ 0.80)  → TRUST : ŷ ± w_normal                    │
        │   YELLOW (0.80<s≤0.98)→ WIDEN : ŷ ± w_widened                  │
        │   RED    (s > 0.98)  → ABSTAIN: "unseen regime, rough alarm"   │
        │   public output = forecast + honest range + flag              │
        └───────────────────────────────┬──────────────────────────────┘
                                         ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  STAGE 6 : EVALUATION  (the money chart)                       │
        │   REGIME-STRATIFIED COVERAGE : normal days vs extreme days     │
        │   baseline 90% → ~60% on extreme ; ours holds ~90% or flags R  │
        │   + matched-band test (proves s ≠ pollution threshold)         │
        └──────────────────────────────────────────────────────────────┘
```

---

## 2. WORKER B — block-by-block spec (fully derived, Parts 1–4)

### B1 — Where distance is measured
- **Primary:** on the model's embedding `h_t` (the "notepad"), because the forecast `ŷ = g(h_t)` is a function of the notepad ALONE. Two days with the same notepad are the same day to the model, so novelty-relative-to-the-model must be a function of `h_t`. (Part 2)
- **Baseline + safety net:** also compute distance on raw features → `s_raw`. Report as a trivial baseline; also take `max(s_raw, s_embed)` because the notepad can "hallucinate normality" on wild inputs, and raw space cannot. (Part 2, §2.8)

### B2 — Per-regime centres (not one global centre)
Delhi is ≥3 clouds (monsoon / winter-inversion / pre-monsoon dust). One global `μ` lands in a physically impossible "average day." So cluster into `k` regimes (k-means on the notepad, or split by season) and keep `μ_k, Σ_k` per regime. (Part 3, Step 13)

### B3 — Plain covariance **[v3]**
- `μ_k, Σ_k` = plain mean and covariance of **all** train hours (2023–24) of season k, for the embedding and for the raw features. MCD 75% + Ledoit–Wolf is **not used**.
- Why (figures/shape_compare.png, all 6 imputation runs): with MCD + LW, ~72% of calibration hours turned Yellow, and on test the Red hours had *smaller* errors than Green hours (Red ÷ Green error 0.70–0.84×). With the plain shape: 37% Yellow and Red ÷ Green error 1.18–1.48× on test, so Red still marks the risky hours.
- Cost: the condition number is higher (embedding ≈ 2,500, raw ≈ 35,000). The check limit is 100,000 (see §4).

### B4 — The distance **[v3]**
`d(x) = sqrt( (h−μ_s)ᵀ Σ_s⁻¹ (h−μ_s) )`, where s = the season of the hour (from the calendar, an input, never the target).
"How far is this hour from the normal hours of **its own season**?"

### B5 — Distance → score (distribution-free) **[v3]**
**Do NOT** convert via χ² — our data isn't Gaussian, the χ² tail is anti-conservative (nominal 1% → ~4–8% flags). Instead use the **empirical rank** against the **train hours of the same season** (2023–24), for `s_embed` and for `s_raw`. Invariant to the true (unknown) distribution's shape. (Part 1 fix; Part 4, Steps 2–3)
Note: calibration 2025 differs from train 2023–24, so more 2025 hours rank high: Red is 3.2–4.6% on test, not 2%.

### B6 — The conformal p-value (the guarantee) **[v3]**
`p_t = (1 + #{train hours of the same season with d ≥ d_t}) / (n + 1)`
The **+1 = counting yourself**. Gives an **exact, finite-sample, distribution-free** bound: flag at level α ⇒ ≤ α of genuinely-normal hours flagged, *for hours exchangeable with train*. Proof = rank is uniform under exchangeability; no bell curve anywhere. (Part 4, Steps 4–6)
**Hard floor:** smallest p is `1/(n+1)` → need `n ≥ 1/α − 1` reference hours per season (each season has ≥ 18,000 train hours).

### B7 — Temporal smoothing **[v3: removed]**
No `β` smoothing. The label comes straight from the fused score of each hour, so it is the same for every horizon.

### B8 — Seasonal calibration
Compare a November day only against other November days, else the detector becomes a 3-month-long "it's winter" calendar. Do it via per-season calibration sets (+ cyclic day-of-year features). (Part 4, Step 8)

### B → output
- `s` = fused score = `max(s_raw, s_embed)` ∈ [0,1] **[v3]** (no smoothing)
- `k` = label GREEN (s ≤ 0.80) / YELLOW (≤ 0.98) / RED, cut-points fixed before test, depending on **features only** (never on true AQI).

---

## 3. WORKER C — stub (to be derived in Parts 5–7)

- **C1–C2:** nonconformity score on calibration residuals; CQR-style width because AQI error is heteroskedastic (bigger errors on dirty days) → constant-width intervals are wrong. *(Part 6)*
- **C3 — One group [v3]:** the CQR scores of **all** calibration hours form one group (no Mondrian split). Reason: calibration (May–Aug 2025) has no winter or post-monsoon hours, so a per-season quantile cannot be made for those seasons. Worker B's label still picks the range: Green → base range, Yellow → wider range (same group, higher level), Red → abstain.
- **C4 — Adaptive update [v3]:** online update for slow drift, step size **γ = 0.001**. γ was picked on the calibration split only (odd weeks = stored scores, even weeks walked): coverage closest to 90%; gaps within 0.5 points count as a tie and the narrower range wins. Candidates 0 · 0.001 · 0.002 · 0.005 · 0.01 · 0.02 · 0.05; every horizon wanted 0–0.002 (figures/gamma_compare.png, gamma_horizon_check.png, gamma_station_check.png). *(Part 8)*
- **output:** width `w` → sent to Decision Layer.

---

## 4. Design constraints & guardrails (must-haves)

| Guardrail | Rule | Source |
|---|---|---|
| **Smallest-group size** | governs BOTH max notepad dimension (`n ≥ 10·d`) AND finest alarm level (`1/(n+1)`). Compute in week 1. | P3 S8 · P4 S5 |
| **Condition number** | print for every `Σ_k`; **[v3]** >100,000 → stop (plain covariance reaches ≈ 2,500 embedding, ≈ 35,000 raw) | P3 S7 |
| **Retrain lock** | notepad meaning changes on retrain → recompute μ_k, Σ_k, calibration distances; version-tag them together | P2 §2.8 |
| **Label uses X only** | regime label must never peek at true AQI | P4 S10 |
| **Thresholds on calib** | G/Y/R cut-points chosen on calibration before touching test | P4 S10 |

---

## 5. Validation plan (Stage 6)

1. **Regime-stratified coverage** (headline): coverage measured separately on normal vs extreme days. Baseline drops below nominal on extreme; ours holds or abstains.
2. **Matched-band test** (kills "it's just a PM2.5 threshold"): bin test days by PM2.5; inside each band (pollution ~constant) show `s` still predicts large model error (AUROC ≫ 0.5). If it fails, s really is a pollution meter — find out now. (Part 4, Step 9)
3. **Baseline ladder:** PM2.5 threshold · "is it November?" calendar rule · raw-Mahalanobis · notepad-Mahalanobis (ours).
4. **Ablations:** raw vs Yeo-Johnson vs rank-Gaussian transform; single-centre vs multi-centre; χ² vs empirical rank; which embedding layer.

---

## 6. Defense one-liners

- **Non-Gaussian?** "Mahalanobis needs only finite second moments (E[d²]=d for any distribution). Normality is only needed for the χ² p-value, which we never use — we calibrate by empirical rank. Non-Gaussianity costs **power**, which we measure, not **validity**, which exchangeability guarantees."
- **Why embeddings?** "The forecast is a function of the embedding alone, so two inputs with the same embedding are indistinguishable to the model — novelty relative to the model must be a function of that embedding."
- **Just detecting pollution?** "Extreme ≠ unfamiliar. We predict model *error*, not pollution *level*, and prove it: within fixed PM2.5 bands `s` still separates big-error from small-error days."
- **Why not a fudge multiplier on the interval?** "An ad-hoc OOD multiplier destroys the coverage guarantee. Our Yellow range is a conformal quantile at a higher level, so it keeps it." **[v3]**

---

*Next: Parts 5–7 fill in Worker C (rank argument → CQR → one group + adaptive γ, see §3 [v3]), Part 8 covers drift, multi-horizon joint coverage, and abstention selection-bias.*
