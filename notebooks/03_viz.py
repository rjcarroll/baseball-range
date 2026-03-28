# %% [markdown]
# # 03 — Visualizations
#
# Build the three main Plotly figures and export as HTML.

# %%
import sys
sys.path.insert(0, "../src")

import pandas as pd
import numpy as np
from baseball_range.data import load_cf_opportunities, add_player_names
from baseball_range.model import fit_all, results_to_df, compute_all_stats
from baseball_range.viz import (
    plot_player_range, plot_range_comparison, plot_rankings,
    plot_opportunity_rankings, plot_spectacular_zone,
)

# %%
# Load fitted results (or refit if needed)
import pathlib

if pathlib.Path("../data/player_ranges.parquet").exists():
    # Load pre-fitted parameters and reconstruct PlayerRange objects
    # For a quick version, re-fit from scratch:
    pass

df = load_cf_opportunities("../data")
df = add_player_names(df)
results = fit_all(df, min_opportunities=75, n_boot=300, seed=42)

# %%
# ── Figure 1: Heatmap for the top-ranked CF ───────────────────────────────────
best = results[0]
fig1 = plot_player_range(best, tau_h=5.0)
fig1.show()
fig1.write_html("../docs/fig_heatmap.html")

# %%
# ── Figure 2: Comparison of top 10 CFs ───────────────────────────────────────
fig2 = plot_range_comparison(results, tau_h=5.0, max_players=10)
fig2.show()
fig2.write_html("../docs/fig_comparison.html")

# %%
# ── Figure 3: Rankings by ellipse area ───────────────────────────────────────
fig3 = plot_rankings(results, tau_h=5.0, top_n=20)
fig3.show()
fig3.write_html("../docs/fig_rankings.html")

# %%
# ── Figures 4 & 5: Derived statistics ────────────────────────────────────────
# Load pre-computed stats if available, otherwise recompute.
import pathlib

if pathlib.Path("../data/player_stats.parquet").exists():
    stats = pd.read_parquet("../data/player_stats.parquet")
else:
    stats = compute_all_stats(results, df, catch_threshold=0.80, n_se=1.0)

# %%
# ── Figure 4: Rankings by opportunity-weighted range ─────────────────────────
fig4 = plot_opportunity_rankings(stats, top_n=20)
fig4.show()
fig4.write_html("../docs/fig_opp_rankings.html")

# %%
# ── Figure 5: Spectacular zone coverage heatmap ───────────────────────────────
fig5 = plot_spectacular_zone(results, tau_h=5.0, catch_threshold=0.80, n_se=1.0)
fig5.show()
fig5.write_html("../docs/fig_spectacular_zone.html")

# %%
# Spectacular play rankings: who performs best in the thin-coverage zone?
spec_ranking = (
    stats.sort_values("spectacular_play_prob", ascending=False)
    [["player_name", "spectacular_play_prob", "opp_weighted_range"]]
    .head(15)
)
print("Top 15 by spectacular play probability:")
print(spec_ranking.to_string(index=False))

# %%
# Sensitivity: how do rankings shift at different hang times?
# Short hang time (2s) rewards players with fast first-step reaction.
# Long hang time (7s) rewards pure top-end speed.
for tau in [2.0, 5.0, 7.0]:
    top5 = [r.player_name for r in results[:5]]
    areas = [r.ellipse_area(tau) for r in results[:5]]
    print(f"τ = {tau}s: " + ", ".join(f"{n} ({a:,.0f} ft²)" for n, a in zip(top5, areas)))
