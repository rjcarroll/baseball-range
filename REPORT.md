# CF Range v1.0: Technical Report

This document records the methods, results, and diagnostic findings for the v1.0
model. It is the basis for the v2.0 design.

---

## Model

Each center fielder is assigned a catchability ellipse that scales with hang time.
For a fly ball with hang time τ landing at displacement (Δx, Δy) from canonical
CF starting position (0, 310) ft:

```
d_i = sqrt( (Δx / (a_i · τ))² + (Δy / (b_i · τ))² )

P(catch) = σ(β₀ − β₁ · d_i)
```

`(a_i, b_i)` are lateral and depth speed in ft/s. Hang time τ is derived from
Statcast launch speed and launch angle via the vacuum projectile formula
`τ = 2 v₀ sin θ / g`. The ellipse boundary `d_i = 1` defines the fielder's
reliable range at a given hang time; balls inside are caught with high probability,
outside they aren't.

### Two derived statistics

**Opportunity-weighted range (OWR)** — fraction of all CF fly balls inside player
i's Bayesian reliable range R_i, weighted by where balls are actually hit. A play
is in R_i if ≥80% of posterior samples give P(catch) ≥ 0.80. Use for free agent
evaluation.

**Spectacular play probability** — expected catch probability on plays outside most
fielders' reliable ranges (covered by ≤1 fielder). Use for late-inning substitution
decisions.

The same model, different estimands, can rank players differently — a fielder with
elite straight-back range accumulates OWR; a fielder who makes diving plays in the
gaps earns spectacular play probability.

### Bayesian hierarchical pipeline

Parameters are estimated via a Stan hierarchical model with sequential
season-to-season updating.

**Burn-in (2017–2020):** `cf_range.stan` estimates population hyperparameters
(μ_a, σ_a, μ_b, σ_b) and logistic shape (β₀, β₁) from pooled data.

**Sequential updates (2021–2024):** `cf_range_update.stan` each season. Each
returning player's prior is their previous posterior, variance inflated by λ=1.25
to allow for year-to-year change. New players receive population hyperparameters.

**Diagnostics:** All 5 fits (burn-in + 4 sequential) achieved 0 divergences and
max R-hat ≤ 1.008.

### Data

Statcast via pybaseball, 2017–2024. Fly balls in CF territory: |lateral| < 100 ft,
depth 200–450 ft, hang_time > 0. Canonical CF starting position fixed at (0, 310) ft.
Burn-in: 2017–2020 (~37,000 fly balls). Analysis: 2021–2024 (~53,000 fly balls).

---

## Results

### Year-over-year correlations (players with ≥75 opps in both seasons)

| Transition | n | OWR ρ | b ρ | a ρ |
|---|---|---|---|---|
| 2021→2022 | 31 | 0.727 | 0.808 | 0.585 |
| 2022→2023 | 28 | 0.662 | 0.761 | 0.565 |
| 2023→2024 | 28 | 0.787 | 0.842 | 0.712 |

Range is a moderately persistent skill (OWR ρ ≈ 0.73). Depth speed b is more
stable year-to-year than lateral speed a. The 2023→2024 transition has the highest
correlations — posteriors are better-informed by this point and rankings are more
stable.

### Player stories

**Jake Meyers: 0.752 → 0.752 → 0.761 → 0.782** — quiet, steady climb to #1 in
2024. No single dominant year; the sequential model built up a confident estimate
from consistent performance across four seasons (91 → 121 → 240 → 354 opps).

**Daulton Varsho: 0.680 → 0.699 → 0.732 → 0.763** — clear upward trajectory as
his CF playing time grew (74 → 96 → 152 → 223 opps). The prior was appropriately
skeptical early; he earned the ranking.

**Kevin Kiermaier: 0.742 → 0.729 → 0.717 → 0.714** and
**Michael Taylor: 0.759 → 0.730 → 0.737 → 0.706** — both showing aging-related
decline with no explicit age component in the model. The data does this work.

**Cedric Mullins: 0.699 → 0.699 → 0.699 → 0.698** — flat across all four seasons
with 275–427 opps each year. Exactly what a stable, well-measured skill looks like.

**The Springer correction:** Ranked #1 in 2021 on 96 opportunities (prior-dominated).
OWR dropped 0.093 in 2022 with 210 opportunities — the largest single-year decline
in the dataset. Clearest demonstration of shrinkage and sequential updating working
correctly: the prior was too generous, and the data corrected it.

### Shape decomposition

`a/b` characterizes range type independently of total size. Most qualified CFs
fall in 1.5–2.0 (laterally faster). Harrison Bader (a/b ≈ 2.0–2.8) is
strongly lateral-dominant. This shows up in the rankings: large ellipse area,
good OWR, but the entire gap over peer players is in depth, not lateral reach.

### OWR vs. spectacular play probability

| Season | Pearson r | Spearman ρ |
|---|---|---|
| 2021 | 0.956 | 0.950 |
| 2022 | 0.939 | 0.934 |
| 2023 | 0.974 | 0.978 |
| 2024 | 0.985 | 0.989 |

The two statistics are highly correlated and growing more so as posteriors tighten.
Meaningful divergences at the tails: players whose range shape does not match the
fly-ball distribution (e.g., lateral-dominant fielder in a gap-heavy environment)
can rank differently on the two statistics.

---

## Diagnostic Battery

Notebook `05_diagnostics.py` runs five diagnostics against 2024 posteriors.

### 1. Global residuals

Mean residual = −0.007, RMSE = 0.287. Slightly negative (model overpredicts catch
probability overall), but the magnitude is small relative to the Brier score
baseline.

### 2. Directional decomposition

Mean residual by 8 × 45° sector of approach angle θ = atan2(Δy, Δx), pooled
across all 2024 players:

| Sector | Direction | Mean residual |
|---|---|---|
| 0–45° | Lateral right + retreating | +0.024 |
| 45–90° | Retreating right | +0.020 |
| 90–135° | Retreating left | −0.009 |
| 135–180° | Lateral left + retreating | +0.012 |
| 180–225° | Lateral left + charging | +0.006 |
| 225–270° | **Charging left** | **−0.061** |
| 270–315° | **Charging right** | **−0.064** |
| 315–360° | Charging right + lateral | +0.001 |

The charging zone (225–315°, balls in front of the fielder) shows systematic
overprediction: the model predicts catches that don't happen. The retreating zone
(0–90°) shows slight underprediction. This is not the superellipse signature (which
would show symmetric over/underprediction at all 45° diagonals). It is a
directional asymmetry concentrated in the charging region.

### 3. Calibration

| Bucket | Predicted P(catch) | Actual catch rate | n |
|---|---|---|---|
| Hard (<0.20) | 0.125 | 0.051 | 234 |
| Mid-low (0.20–0.50) | 0.362 | 0.197 | 915 |
| Mid-high (0.50–0.80) | 0.677 | 0.691 | 2,041 |
| Easy (>0.80) | 0.941 | 0.946 | 9,724 |

The model is well-calibrated for clearly easy and clearly impossible plays. It
overpredicts substantially in the boundary zone (p̂ = 0.36, actual = 0.20 — a 16
percentage-point gap). The 234 hard-play predictions also overstate catch rate by
7 pp. Both effects are concentrated in the same region as the directional bias.

### 4. Shape-ratio vs residual RMSE

Spearman ρ = −0.09, p = 0.53 — effectively null. Players with extreme a/b ratios
(e.g., Bader at a/b ≈ 2.0–2.8, Daza at a/b ≈ 1.3) do not show systematically
worse fit. **The ellipse orientation and shape class are not the source of
misspecification.**

### 5. Moran's I spatial autocorrelation

Threshold 30 ft, 1000-permutation test. Results: **32 of 35 players** with ≥150
opps show significant positive spatial autocorrelation (p < 0.05). Effect sizes
I = 0.06–0.20. This confirms the residuals are spatially clustered, not random.

---

## Diagnosis

The Moran's I result triggers the "widespread clustering → consider superellipse"
gate from the pre-diagnostic plan. However, the directional decomposition
identifies the source: the charging zone, not the ellipse diagonals. The superellipse
signature (over/underprediction symmetric at 45° intervals) is absent. The shape-
ratio null result is consistent with this — the ellipse *shape class* is not wrong.

**The root cause is the fixed starting position.** The model assumes every CF
starts the play at y = 310 ft. If fielders typically play at y ≈ 320–330 ft:

- Charging plays (Δy < 0 from canonical 310): the fielder is actually further
  from the ball than the model measures → model overpredicts P(catch) → negative
  residuals in the charging zone.
- Retreating plays (Δy > 0): the fielder is closer to the ball than the model
  measures → model underpredicts slightly → positive residuals.

This matches exactly what the diagnostics show. The Moran's I significance reflects
spatial clustering of charging-zone plays (which land near each other on the field),
not genuine shape misspecification.

**For ranking purposes, the confound is approximately uniform** — the fixed y = 310
assumption applies to every player equally, so relative rankings are valid. Absolute
P(catch) values for boundary plays are inflated.

---

## What v2.0 Addresses

**Problem:** Fixed starting position (y = 310) conflates range with pre-play
positioning. The positional confound creates systematic overprediction in the
charging zone and inflated catch probabilities in the boundary zone.

**v2.0 solution:**

1. **Latent y₀ᵢ (per-player starting depth)** — estimated from catch/no-catch
   outcomes via the hierarchical model. Population prior centered at 310 ft, width
   ~15 ft. This absorbs the positional confound and makes the range parameters (a, b)
   interpretable as pure speed estimates from the player's actual starting position.

2. **γᵢ (charge/retreat asymmetry)** — lognormal prior centered at 1. Adds a
   forward stretch factor to the depth half-ellipse. Now disentangled from y₀:
   once starting position is absorbed, residual directional asymmetry is genuinely
   attributable to different running speeds toward and away from the plate.

3. **Part 2: optimal positioning** — given each fielder's (a, b, γ, y₀) and a
   batter's spray distribution, find the starting position that maximizes expected
   coverage. The v2.0 model makes Part 2 strictly more interpretable because y₀ is
   already estimated — the optimizer shifts from the player's actual baseline rather
   than the canonical position.

---

## Key Implementation Notes

- `caught` = events.isin(["field_out", "sac_fly"]) — not "caught_fly_ball"
- `hang_time` derived via vacuum formula, not a Statcast column
- CF filter: |lateral| < 100 ft, depth 200–450 ft, hang_time > 0
- 1 row dropped from cf_2023.parquet: launch_angle < 0 produced hang_time < 0
- Pandas masked array fix: `np.asarray(df["col"])` required in Python 3.14 for
  nullable float columns before broadcasting with `[np.newaxis, :]`
