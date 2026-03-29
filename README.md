# Who is the Best Defensive Center Fielder?

An interesting question — but "best" depends on what you're trying to decide.
This project builds a parametric range model fit to Statcast fly-ball data and
derives two statistics from it, each designed for a different decision. The
same model, asked two different questions, can rank players differently.

Each fielder gets a catchability ellipse that scales with hang time; two
parameters — lateral and depth speed in ft/sec — define how far they can run
in each direction.

---

## The model

**v2.0** — current model. Four per-player parameters.

For a fly ball with hang time τ landing at displacement (Δx, Δy) from the
canonical CF reference point (0, 310) ft:

```text
dy_adj   = Δy − y0_offset_i
w        = sigmoid(−dy_adj / 5)      # charging weight: ≈1 in front, ≈0 behind
b_eff    = b_i · exp(log_γ_i · w)   # effective depth speed

d_i = sqrt( (Δx / (a_i · τ))² + (dy_adj / (b_eff_i · τ))² )

P(catch) = sigmoid(β₀ − β₁ · d_i)
```

`(a_i, b_i)` are lateral and depth speed in ft/sec. `y0_offset_i` is the
player's inferred starting depth offset from 310 ft (estimated from outcome
data, not pre-play tracking). `log_γ_i` is a charge/retreat asymmetry factor:
`γ > 1` means the player covers more ground going in than going back.

The sigmoid blend (width 5 ft) ensures HMC differentiability. At |dy_adj| > 15
ft the blend is >95% saturated, so the approximation is accurate for all but
the most precisely-measured short-fly plays.

**Hang time** is not a Statcast column. It is derived from `launch_speed` and
`launch_angle` using the vacuum projectile formula:

```text
τ = 2 · v₀ · sin(θ) / g
```

---

### v1.0 model (archived)

The original model fixed starting position at (0, 310) ft and used only
`(a_i, b_i)`:

```text
d_i = sqrt( (Δx / (a_i · τ))² + (Δy / (b_i · τ))² )
```

Diagnostics on 2021–2024 found systematic overprediction in the charging zone
(sectors 225–315°, mean residual −0.061/−0.064) and 32/35 qualified players
showing significant spatial autocorrelation (Moran's I, p < 0.05). Root cause:
fielders typically play at y ≈ 320–330 ft, not 310 ft — the fixed origin
conflated range with pre-play positioning. Shape-ratio vs. RMSE correlation was
null (ρ = −0.09), ruling out ellipse misspecification as the cause. Full
diagnostic record in `REPORT.md`. v1.0 posteriors archived in `data/v1/`.

---

## Reading the parameters

`a` and `b` are speeds in ft/s, but several derived quantities make them
easier to interpret.

**Ellipse area** at hang time τ is π · a · b · τ². At τ = 5 seconds — a
typical long fly ball — the 2021 median qualified CF covers roughly 23,300 sq ft.
The best cover ~29,700 sq ft, about 27% more ground. The gap between the best
and a league-average CF is on the order of 6,000 sq ft, comparable in size to
a baseball diamond.

**Marginal plays per season.** The difference in `opp_weighted_range` between
two players, multiplied by the ~425 CF fly-ball opportunities per team slot per
season, gives a concrete defensive volume. In 2021, Bader's range advantage over
the median CF translates to roughly +34 plays per season; Taylor's to +26.
A 0.03 difference in `opp_weighted_range` ≈ ~13 additional plays per season.

**Shape ratio a/b** characterizes the *type* of range independent of total size.
Most qualified CFs fall in the range 1.5–2.0 (laterally faster than depth-wise),
but the spread is meaningful. Naquin (a/b ≈ 2.2) is strongly laterally-oriented;
Daza (a/b ≈ 1.3) is relatively balanced with depth emphasis. Crucially, total
ellipse area and `opp_weighted_range` can diverge sharply based on shape: Naquin
has a larger ellipse than Taylor at τ = 5s, but a below-median `opp_weighted_range`
because his range shape does not match where CF fly balls actually land. A
lateral-dominant fielder covers a lot of ground that balls rarely reach.

**Bader vs. Bradley (2021)** illustrates the shape decomposition cleanly.
Both have nearly identical lateral reach (~125 ft at τ = 5s). The entire
difference in their ellipses — Bader is 13.6% larger — comes from depth: Bader
reaches 75 ft deep vs. Bradley's 65 ft. Ten feet of additional depth range,
accumulated over a season, accounts for roughly +26 plays per season.

---

## Two derived statistics

The parameters are the model. The decision determines which statistic to
derive from them.

### Reliable range

The set of landing spots where the fielder makes the play with high probability
*and* high confidence:

```text
R_i = { (x, y) : P(catch | a_i, b_i) ≥ 0.80 in ≥ 80% of posterior samples }
```

Using the full posterior rather than a point estimate makes `R_i` properly
account for estimation uncertainty. Low-opportunity players get a smaller
reliable range because their posterior is wider.

**Opportunity-weighted range** — the preferred single-number ranking — weights
`R_i` by where balls are actually hit:

```python
def opportunity_weighted_range(player, all_fly_balls):
    # A play is reliably covered if ≥80% of posterior samples give P(catch) ≥ 0.80
    in_range = fraction_above_threshold(player.posterior, fly_ball) >= 0.80
    # Fraction of all CF fly balls reliably covered
    return in_range.mean()
```

### Spectacular play probability

Define the spectacular zone `S` as plays outside most fielders' reliable
ranges — where the best CFs make plays others can't:

```python
def compute_spectacular_zone(all_players, fly_balls, max_coverage=1):
    # How many fielders reliably cover each play?
    coverage = sum(reliable_range_indicator(p, fly_balls) for p in all_players)
    # Spectacular = covered by at most 1 fielder
    return coverage <= max_coverage

def spectacular_play_prob(player, fly_balls, spectacular_mask):
    hard_plays = fly_balls[spectacular_mask]
    return catch_prob(player.a_mean, player.b_mean, hard_plays).mean()
```

### Why they can rank players differently

A fielder with elite straight-back range racks up `opp_weighted_range`. A
fielder who makes diving plays on balls in the gaps earns `spectacular_play_prob`.
Same model, different estimands, different decisions:

| Decision | Statistic |
| --- | --- |
| Free agent evaluation | Largest `opp_weighted_range` |
| Late-inning substitution | Highest `spectacular_play_prob` |
| Park fit | Shape of R_i, not just size |

---

## Bayesian pipeline

Parameters are estimated via a Stan hierarchical model with sequential
season-to-season updating. Each player has a full posterior over `(a_i, b_i)`,
and `R_i` becomes a random set — the credible band around the boundary captures
uncertainty about which set it is, not just a scalar interval.

### Two Stan models

`stan/cf_range_v2.stan` — **burn-in**. Fit once on 2017-2020 data. Estimates
population hyperparameters `(μ_a, σ_a, μ_b, σ_b, μ_y0, σ_y0, μ_log_γ,
σ_log_γ)` and logistic shape `(β₀, β₁)`. Returns per-player posteriors for
all four parameters.

`stan/cf_range_update_v2.stan` — **sequential update**. All eight per-player
prior parameters are passed as data rather than estimated. `β₀, β₁` are fixed
at burn-in posterior means. Run on each subsequent season.

### Sequential updating

Each player's prior for a new season is their previous season's posterior,
inflated by a factor λ to allow for year-to-year change:

```python
prior_sigma = lam * posterior_sd   # λ = 1.25 (one off-season)
```

New players receive the population hyperparameters as their prior. The result
is a proper Bayesian update: players with few observations are pulled toward
the population mean (shrinkage), while players with many observations are
barely affected.

**Lambda schedule:** λ = 1.25 for all annual transitions. Floors prevent
priors collapsing to point masses: σ_y0 ≥ 0.5, σ_log_γ ≥ 0.05.

---

## Part 2: Optimal positioning

Once each player's `(a_i, b_i, y0_offset_i, γ_i)` are estimated, the natural
follow-on question is not just *who* is best but *where should you put him*
against a given hitter.

For each batter, Statcast gives us a spray distribution over CF fly-ball
landing spots. Given a fielder's parameters and that distribution, the optimal
pre-pitch position is:

```text
(x̂₀, ŷ₀) = argmax_{x₀, y₀} E[coverage | x₀, y₀, batter_dist, a_i, b_i, γ_i]
```

The v2.0 model makes this strictly more interpretable than v1.0: `y0_offset_i`
is already estimated, so the positioning optimizer shifts from the player's
inferred actual baseline rather than the canonical 310 ft. A 2D grid search or
`scipy.optimize.minimize` over a ~25×25 ft grid is tractable. The answer
depends on both the fielder's range shape and the batter — a depth-dominant
fielder may shade differently than a lateral-dominant peer facing the same pull
hitter.

---

## Further extensions

### Superellipse boundary

The v2.0 diagnostics ruled out shape misspecification as the source of spatial
clustering (shape-ratio vs. RMSE ρ = −0.09). But the deeper question — whether
the ellipse (p = 2) is the right shape class — is still worth testing after the
positional confound is absorbed. The superellipse (Lamé curve)

```text
d = ( |Δx / (a·τ)|^p + |Δy / (b·τ)|^p )^(1/p)
```

generalizes the ellipse (p = 2) toward a rectangle (p → ∞). The exponent p
estimates directional coupling. The Stan change is one line; the empirical test
is to rerun Moran's I on v2.0 residuals and check whether clustering disappears.
If it does, the ellipse is sufficient. If it persists with a different spatial
pattern (symmetric at 45° diagonals rather than the charging zone), the
superellipse earns its keep.

### Diagonal asymmetry

A fourth parameter could capture left-right interaction: a correlation
coefficient ρ in the distance metric yields a tilted ellipse

```text
d² = (1/(1−ρ²)) · [ (Δx/a)² − 2ρ(Δx/a)(Δy/b) + (Δy/b)² ]
```

whose principal axes rotate away from the field coordinates. `(a, b, γ)` and
`(a, b, ρ)` capture different shapes of asymmetry and are not nested; a full
treatment needs all four parameters. The right test: fit v2.0, decompose
residuals by quadrant, and ask whether systematic diagonal structure persists
for high-data players before adding the estimation burden.

### Age curve on the prior mean

The sequential update currently holds the prior mean fixed at the previous
season's posterior mean — the best guess for next year is "same as last year."
A natural refinement is to shift the mean along a population-level age curve:

```text
prior_mu_a(t+1) = posterior_mean(t) + Δf(age)
```

where `f(age)` is a smooth function (quadratic or spline) peaking in the late
20s, estimated hierarchically from all players simultaneously. A 27-year-old's
prior for next year is slightly better than their current posterior; a
33-year-old's is slightly worse. The variance inflation (λ) stays the same —
that captures off-season uncertainty independent of age trend. The age curve
itself is a population-level parameter estimated from the data, which is exactly
the kind of thing a hierarchical model handles well and per-player regression
cannot (too few seasons per player to estimate aging individually).

### Measurement uncertainty and quadrature

Landing coordinates in Statcast carry measurement error (~1–2 ft from HawkEye
triangulation) and physical spread (drag variation, wind). Treating each
observation as a precise (Δx, Δy) point is an approximation. The proper
likelihood marginalizes over the landing distribution:

```text
P(catch | player, obs) = ∫ P(catch | x, y) · p(x, y | recorded) dx dy
```

With a Gaussian uncertainty kernel, this integral can be approximated
accurately using Gaussian quadrature — 5–9 evaluation points per observation,
negligible runtime cost. The effect: smoother likelihood, reduced sensitivity to
outlier observations, and honest treatment of plays landing near the range
boundary.

### Environmental covariates

Wind is the largest omitted variable — a headwind meaningfully changes where a
fly ball lands relative to still-air prediction. The right correction is at
the trajectory level: adjust the observed (Δx, Δy) for wind before computing
`d_i`, so the fielder's ellipse is estimated against wind-corrected displacements.
Temperature (air density → drag) and sun angle (execution failure on catchable
balls near the sun) are secondary effects but followable from the same framework.

### Hitter and pitch covariates

Exit velocity distribution matters through hang time: hard-hit balls at a given
distance have shorter τ, compressing the fielder's window. This flows through
the model correctly via the observed τ. Pitch type shapes the batted-ball
distribution upstream of the observation and is already conditioned out by
modeling displacement directly. The more substantive covariate is the
**opposing lineup** — a player's observed range in a season depends partly on
who they faced. The current model is already robust to this by conditioning on
displacement, but explicit lineup adjustment would be needed for a true
fielder-vs-fielder comparison across different team contexts.

---

## Figures

Interactive Plotly figures from the Bayesian pipeline, exported to `docs/`:

1. **Posterior ellipse** — posterior-mean heatmap + semi-transparent sample ellipses forming a credible band (v2.0: asymmetric half-ellipse, marker at inferred starting position)
2. **Bayesian rankings** — horizontal bar chart with ±1 SD credible interval error bars
3. **Season evolution** — monthly snapshots through a season with narrowing credible bands (April→September)
4. **Prior vs. posterior** — two-panel density diagnostic; shrinkage visible for thin-data players
5. **Calibration** — actual catch rate vs. predicted P(catch), 10 quantile bins (v1.0 diagnostic)

---

## Data

Statcast via `pybaseball`. Seasons 2017-2024. Filtered to fly balls in CF
territory (|lateral| < 100 ft, depth 200–450 ft from home plate). Canonical
CF starting position fixed at (0, 310) ft.

Seasons 2017-2020 serve as burn-in for the Bayesian model. Analysis window
is 2021-2024.

---

## Quickstart

```bash
cd ~/portfolio/baseball-range
pip install -e ".[bayes]"

# 1. Pull Statcast data (~30-80 min per season, then cached)
#    open notebooks/01_data.py

# 2. Run Bayesian pipeline (~1-2 hr first run, loads from cache after)
cd notebooks
caffeinate -i python3 04_bayes.py 2>&1 | tee ../logs/bayes_run.log
```

CmdStan must be installed: `python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`

---

## Directory layout

```text
src/baseball_range/
  data.py      Statcast pull, coordinate transforms, hang_time derivation, CF filter
  model.py     Ellipse model math, PlayerRange dataclass
  bayes.py     Bayesian pipeline: BayesPlayerRange, PopulationHyperparams,
               fit_burnin, fit_season, run_sequential_pipeline,
               fit_season_evolution, Bayesian stats functions (v2.0 distance formula)
  viz.py       Plotly figures including asymmetric half-ellipse rendering (v2.0)
notebooks/
  01_data.py         Pull and inspect Statcast data (Jupytext format)
  04_bayes.py        Full Bayesian pipeline, sequential updates, figures
  05_diagnostics.py  v1.0 diagnostic battery (5 tests, motivated v2.0 design)
stan/
  cf_range_v2.stan         Burn-in model (v2.0: adds y0_offset, log_gamma)
  cf_range_update_v2.stan  Sequential update model (v2.0: all 4 priors as data)
  cf_range.stan             v1.0 burn-in (archived)
  cf_range_update.stan      v1.0 update (archived)
data/
  cf_{year}.parquet  Cached Statcast pulls (not tracked)
  v1/                v1.0 posteriors and stats (archived)
docs/          Exported HTML figures
REPORT.md      v1.0 technical record: methods, results, diagnostics, v2.0 design
logs/          Run logs (not tracked)
```
