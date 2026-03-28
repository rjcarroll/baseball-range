# Who is the Best Defensive Center Fielder?

A parametric range model fit to Statcast fly-ball data. Each fielder gets a
catchability ellipse that scales with hang time; two parameters — lateral and
depth speed in ft/sec — define how far they can run in each direction.

The key idea from the interview answer that motivated this project: **the same
model supports two different rankings that answer two different questions.**
Free agent evaluation calls for one stat; late-inning substitution calls for
another.

---

## The model

For a fly ball with hang time τ landing at displacement (Δx, Δy) from the
fielder's canonical starting position:

```text
d_i = sqrt( (Δx / (a_i · τ))² + (Δy / (b_i · τ))² )

P(catch) = sigmoid(β₀ − β₁ · d_i)
```

`(a_i, b_i)` are lateral and depth speed in ft/sec. Balls inside the ellipse
(`d_i < 1`) are caught with high probability; outside, they aren't. Hang time
scaling encodes the physics — 150 feet is a different play with 6 seconds of
air time versus 2.

Parameters are estimated per player via MLE (scipy), with bootstrap resampling
for uncertainty on `(a_i, b_i)`.

---

## Two derived statistics

The parameters are the model. The decision determines which statistic to
derive from them.

### Reliable range

The set of landing spots where the fielder makes the play with high probability
*and* high confidence, from a common canonical starting position:

```text
R_i = { (x, y) : P(catch | conservative params) ≥ 0.80 }
```

"Conservative params" means `a_i − 1·SE`, `b_i − 1·SE` — shrinking the
ellipse by one bootstrap standard error, so the region is reliable both in
probability and in estimation certainty.

**Opportunity-weighted range** — the preferred single-number ranking — weights
`R_i` by where balls are actually hit:

```python
def opportunity_weighted_range(player, all_fly_balls):
    # Is each observed CF fly ball inside this player's reliable range?
    in_range = catch_prob(conservative_params, fly_ball) >= 0.80
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
    return catch_prob(player.params, hard_plays).mean()
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

## Figures

Five interactive Plotly figures, exportable to HTML:

1. **Catch probability heatmap** — single player, showing the ellipse and
   probability gradient across the outfield
2. **Range comparison** — overlaid 50%-boundary ellipses for multiple CFs
3. **Rankings by ellipse area** — with bootstrap ±1 SE error bars
4. **Rankings by opportunity-weighted range** — the preferred ranking stat
5. **Coverage heatmap** — how many fielders have reliable range at each
   field location; dark spots are the spectacular zone

---

## Data

Statcast via `pybaseball`. Seasons 2021–2024. Filtered to fly balls in CF
territory (|lateral| < 100 ft, depth 200–450 ft from home plate). Canonical
CF starting position fixed at (0, 310) ft.

**Hang time** is not a Statcast column. It is derived from `launch_speed` and
`launch_angle` using the vacuum projectile formula:

```text
t = 2 · v₀ · sin(θ) / g
```

where v₀ is launch speed converted to ft/s and θ is launch angle in radians.
Drag reduces range more than hang time, so this is a reasonable proxy for how
long the fielder has to reach the ball.

**Limitation:** Statcast records fielder identity and landing coordinates, but
not fielder position at pitch time. Starting position is fixed at the canonical
value. Positioning variation is real but not identified without per-play
Sprint Speed data.

---

## Bayesian version

A Stan hierarchical model in `stan/cf_range.stan` partially pools `(a_i, b_i)`
across players — valuable for players with fewer opportunities. This produces
full posteriors on each player's ellipse, making `R_i` a *random set*: the
credible band around the boundary is uncertainty about which set it is, not
just a scalar interval.

---

## Quickstart

```bash
cd ~/portfolio/baseball-range
pip install -e ".[dev]"
# open notebooks/01_data.py — pulls Statcast data (~30-80 min, then cached)
# open notebooks/02_model.py — fits model, computes both stats
# open notebooks/03_viz.py — builds and exports all 5 figures
```

## Directory layout

```text
src/baseball_range/
  data.py      Statcast pull, coordinate transforms, CF opportunity filter
  model.py     MLE per player; bootstrap SEs; opportunity_weighted_range,
               spectacular_play_prob, compute_all_stats
  viz.py       5 Plotly figures
notebooks/
  01_data.py   Pull and inspect Statcast data (Jupytext format)
  02_model.py  Fit model, compute both stats, save player_stats.parquet
  03_viz.py    Build and export all figures
stan/
  cf_range.stan  Hierarchical Bayesian version (partial pooling)
data/          Cached Statcast pulls (not tracked)
docs/          Exported HTML figures
```
