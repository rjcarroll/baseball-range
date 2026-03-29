# %% [markdown]
# # 02 — Model Fitting
#
# Estimate per-player (a, b) ellipse parameters via MLE and bootstrap SEs.
# Results are saved to `data/player_ranges.parquet`.

# %%
import sys
sys.path.insert(0, "../src")

import pandas as pd
import numpy as np
from baseball_range.data import load_cf_opportunities, add_player_names
from baseball_range.model import fit_all, results_to_df, compute_all_stats, filter_identified

# %%
df = load_cf_opportunities("../data")
df = add_player_names(df)
print(f"{len(df):,} opportunities, {df['player_id'].nunique()} players")

# %%
# Fit all players with >= 75 opportunities
# Runtime: ~5 min (300 bootstrap samples per player, ~80-100 qualifying players)
results = fit_all(df, min_opportunities=75, n_boot=300, seed=42, n_jobs=4)
print(f"Fitted {len(results)} players")

# %%
# Filter out players where MLE drifted to implausible values (unidentified a or b).
# These are players with few lateral opportunities — the likelihood is flat in the
# a direction so the optimizer wanders to physically impossible speeds.
# The Bayesian pipeline handles these correctly via the population prior.
results_id, results_unid = filter_identified(results)
if results_unid:
    print(f"\nDropped {len(results_unid)} player(s) with unidentified MLE parameters:")
    for r in results_unid:
        print(f"  {r.player_name:<26s}  a={r.a:.1f}  b={r.b:.1f}  n={r.n_opportunities}")
print(f"\n{len(results_id)} players with identified parameters proceeding to rankings.\n")

# %%
rankings = results_to_df(results_id)
print(rankings.head(20).to_string(index=False))

# %%
# Save raw parameter estimates for downstream use
rankings.to_parquet("../data/player_ranges.parquet", index=False)
print("Saved to data/player_ranges.parquet")

# %%
# ── Derived statistics ────────────────────────────────────────────────────────
# Compute both decision-relevant estimands against the full empirical distribution.
#
# opp_weighted_range:  fraction of ALL CF fly balls player i reliably covers
#                      (the ∫_{R_i} f(x,y) integral — preferred for FA evaluation)
# spectacular_play_prob: expected catch probability on plays outside most
#                        fielders' reliable ranges (preferred for late-inning subs)

stats = compute_all_stats(results_id, df, catch_threshold=0.80, n_se=1.0,
                          max_spectacular_coverage=1)
print(stats[["player_name", "opp_weighted_range", "spectacular_play_prob",
             "ellipse_area_5s", "n_opportunities"]].head(20).to_string(index=False))

# %%
stats.to_parquet("../data/player_stats.parquet", index=False)
print("Saved to data/player_stats.parquet")

# %%
# How different are opp_weighted_range rankings vs. ellipse_area rankings?
by_area = stats.sort_values("ellipse_area_5s", ascending=False)["player_name"].tolist()
by_owr  = stats["player_name"].tolist()  # already sorted by opp_weighted_range

rank_shift = {
    name: by_area.index(name) - by_owr.index(name)
    for name in by_owr
}
print("Rank shift (positive = ranked higher by opp-weighted range):")
for name, shift in sorted(rank_shift.items(), key=lambda x: -abs(x[1]))[:10]:
    print(f"  {name:25s}  {shift:+d}")

# %%
# Spot-check a player: look at their catch-probability curve along the depth axis
from baseball_range.model import catch_probability, normalized_distance

def inspect_player(pr, tau_h=5.0):
    depths = np.linspace(200, 450, 200)
    dx = np.zeros_like(depths)
    dy = depths - 310.0  # displacement from canonical CF_Y0=310
    tau = np.full_like(depths, tau_h)
    d = normalized_distance(dx, dy, tau, pr.a, pr.b)
    p = catch_probability(d, pr.beta_0, pr.beta_1)
    return depths, p

# Best and worst by area
best = results_id[0]
worst = results_id[-1]

import plotly.graph_objects as go
fig = go.Figure()
for pr, label in [(best, "best"), (worst, "worst")]:
    depths, p = inspect_player(pr)
    fig.add_trace(go.Scatter(
        x=depths, y=p, mode="lines",
        name=f"{pr.player_name} ({label})",
    ))
fig.update_layout(
    title="Catch probability vs. depth (zero lateral, hang time = 5s)",
    xaxis_title="Depth (ft from home plate)",
    yaxis_title="P(catch)",
    yaxis=dict(tickformat=".0%"),
)
fig.show()
