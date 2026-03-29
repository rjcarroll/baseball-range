# %% [markdown]
# # 04 — Bayesian CF Range Pipeline
#
# Full Bayesian hierarchical estimation with sequential season-to-season updates.
#
# ## Pipeline
# - **Phase 0 (burn-in):** Fit `cf_range_v2.stan` on 2017-2020 data. Estimates
#   population hyperparameters, logistic shape (β₀, β₁), and per-player
#   y0_offset (starting depth) + log_gamma (charge/retreat asymmetry).
#
# - **Phase 1 (sequential updates):** For each season 2021-2024, each returning
#   player's prior is their previous posterior inflated by λ (to allow for
#   year-to-year change). New players receive the population hyperparameters.
#   Fit via `cf_range_update_v2.stan`.
#
# Lambda schedule: λ = 1.25 for all transitions (one off-season of uncertainty).
#
# **Runtime:** ~30-90 min total (4000 MCMC draws × P players per season).
# Cached posteriors in `data/` are used on subsequent runs.

# %%
import sys
sys.path.insert(0, "../src")

import pathlib
import pandas as pd
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

from baseball_range.data import load_cf_opportunities, pull_seasons, add_player_names
from baseball_range.bayes import (
    run_sequential_pipeline,
    bayes_compute_all_stats,
    fit_season_evolution,
    save_posteriors, load_posteriors,
    DEFAULT_LAM_SCHEDULE,
)
from baseball_range.viz import (
    plot_posterior_ellipse, plot_bayes_rankings,
    plot_season_evolution, plot_prior_posterior_update,
    plot_bayes_spectacular_zone, plot_range_trajectories,
)

STAN_DIR = pathlib.Path("../stan")
CACHE_DIR = pathlib.Path("../data")

# %%
# ── Load all seasons ──────────────────────────────────────────────────────────
# Burn-in: 2017-2020   |   Analysis: 2021-2024
# Pull any missing seasons first.

BURNIN_SEASONS = [2017, 2018, 2019, 2020]
UPDATE_SEASONS = [2021, 2022, 2023, 2024]
ALL_SEASONS = BURNIN_SEASONS + UPDATE_SEASONS

# Pull any missing seasons (cached seasons load from parquet, not re-pulled)
pull_seasons(ALL_SEASONS, cache_dir=str(CACHE_DIR))

data_by_season = {}
for season in ALL_SEASONS:
    path = CACHE_DIR / f"cf_{season}.parquet"
    df = pd.read_parquet(path)
    df = add_player_names(df)
    df["game_date"] = pd.to_datetime(df["game_date"])
    data_by_season[season] = df
    print(f"{season}: {len(df):,} opportunities, {df['player_id'].nunique()} players")

# %%
# ── Phase 0 + Phase 1: full sequential pipeline ───────────────────────────────
# Cached after first run. Re-runs are fast (loads .npz / .json).

posteriors_by_season = run_sequential_pipeline(
    data_by_season=data_by_season,
    burnin_seasons=BURNIN_SEASONS,
    update_seasons=UPDATE_SEASONS,
    stan_burnin=str(STAN_DIR / "cf_range_v2.stan"),
    stan_update=str(STAN_DIR / "cf_range_update_v2.stan"),
    lam_schedule=DEFAULT_LAM_SCHEDULE,
    n_chains=4,
    n_samples=1000,
    seed=42,
    cache_dir=str(CACHE_DIR),
)

print(f"\nFit seasons: {sorted(posteriors_by_season.keys())}")
for season, posts in sorted(posteriors_by_season.items()):
    print(f"  {season}: {len(posts)} players")

# %%
# ── Per-season statistics ─────────────────────────────────────────────────────
# Combine all 2021-2024 observations as the empirical distribution for
# opportunity-weighting. This makes the statistic comparable across seasons.

df_analysis = pd.concat(
    [data_by_season[s] for s in UPDATE_SEASONS], ignore_index=True
)

stats_by_season = {}
for season in UPDATE_SEASONS:
    print(f"\n{'─'*60}")
    print(f"Season {season}")
    stats = bayes_compute_all_stats(
        posteriors_by_season[season],
        df_analysis,
    )
    stats_by_season[season] = stats
    print(stats[["player_name", "opp_weighted_range", "spectacular_play_prob_mean",
                 "ellipse_area_5s_mean", "n_opportunities"]].head(15).to_string(index=False))
    stats.to_parquet(CACHE_DIR / f"bayes_stats_{season}.parquet", index=False)
    print(f"Saved to data/bayes_stats_{season}.parquet")

# %%
# ── Key comparison: MLE vs. Bayes (shrinkage visible) ────────────────────────
# Load MLE results (from notebook 02) if available; scatter vs. Bayesian means.

import plotly.graph_objects as go

try:
    mle_stats = pd.read_parquet(CACHE_DIR / "player_stats.parquet")
    bayes_2024 = stats_by_season[2024].rename(columns={
        "a_mean": "a_bayes", "b_mean": "b_bayes", "n_opportunities": "n_bayes"
    })
    merged = mle_stats.merge(
        bayes_2024[["player_id", "a_bayes", "b_bayes", "n_bayes"]],
        on="player_id", how="inner"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["a"], y=merged["a_bayes"],
        mode="markers",
        text=merged["player_name"],
        marker=dict(
            size=np.sqrt(merged["n_opportunities"]) * 1.5,
            color=merged["n_opportunities"],
            colorscale="Blues",
            colorbar=dict(title="n opps"),
            showscale=True,
        ),
        hovertemplate="<b>%{text}</b><br>MLE a: %{x:.1f}<br>Bayes a: %{y:.1f}<extra></extra>",
    ))
    lo = min(merged["a"].min(), merged["a_bayes"].min()) - 1
    hi = max(merged["a"].max(), merged["a_bayes"].max()) + 1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Lateral speed a: MLE vs. Bayesian posterior mean (2021-2024)",
        xaxis_title="MLE â (ft/s)",
        yaxis_title="Bayes posterior mean (ft/s)",
        width=600, height=560,
    )
    fig.show()
    print("Low-opportunity players: shrinkage pulls toward population mean.")
    print("High-opportunity players: Bayes ≈ MLE.")
except FileNotFoundError:
    print("player_stats.parquet not found — run notebook 02 first to compare.")

# %%
# ── Figures ───────────────────────────────────────────────────────────────────
# Focus on 2024 posteriors.

season = 2024
posts_2024 = posteriors_by_season[season]
stats_2024 = stats_by_season[season]

# Sort by opp_weighted_range for figure selection
top_ids = stats_2024.head(3)["player_id"].tolist()
top_players = [p for p in posts_2024 if p.player_id in top_ids]
top_players_sorted = sorted(top_players, key=lambda p: top_ids.index(p.player_id))

# Figure 6: posterior ellipse for best CF
fig6 = plot_posterior_ellipse(top_players_sorted[0], tau_h=5.0)
fig6.show()
fig6.write_html("../docs/bayes_posterior_ellipse.html", include_plotlyjs="cdn")
print("Saved bayes_posterior_ellipse.html")

# Figure 7: Bayesian rankings
fig7 = plot_bayes_rankings(stats_2024, metric="opp_weighted_range", top_n=20)
fig7.show()
fig7.write_html("../docs/bayes_rankings.html", include_plotlyjs="cdn")
print("Saved bayes_rankings.html")

# Figure 7b: Spectacular play rankings
fig7b = plot_bayes_rankings(stats_2024, metric="spectacular_play_prob_mean", top_n=20)
fig7b.show()
fig7b.write_html("../docs/bayes_rankings_spectacular.html", include_plotlyjs="cdn")
print("Saved bayes_rankings_spectacular.html")

# Figure 7c: Spectacular zone coverage heatmap
fig7c = plot_bayes_spectacular_zone(posts_2024, tau_h=5.0)
fig7c.show()
fig7c.write_html("../docs/bayes_spectacular_zone.html", include_plotlyjs="cdn")
print("Saved bayes_spectacular_zone.html")

# Figure 7d: Range trajectories across seasons (connected dot plot)
fig7d = plot_range_trajectories(stats_by_season, min_seasons=3, min_opps=75, top_n=30)
fig7d.show()
fig7d.write_html("../docs/bayes_range_trajectories.html", include_plotlyjs="cdn")
print("Saved bayes_range_trajectories.html")

# %%
# Figure 9: Prior vs. posterior diagnostic (one player, illustration)
# Use the player with fewest opportunities to show shrinkage.
from baseball_range.bayes import build_sequential_priors, PopulationHyperparams
import json

# Load hyperparams from burn-in cache
hp_path = str(CACHE_DIR / "burnin_hyperparams.json")
try:
    with open(hp_path) as f:
        hp_dict = json.load(f)
    hyperparams = PopulationHyperparams(**hp_dict)

    # Pick a low-opportunity player for illustration
    thin_player = min(
        [p for p in posts_2024 if p.n_opportunities < 60],
        key=lambda p: p.n_opportunities,
        default=top_players_sorted[-1]
    )

    # Reconstruct the prior used for this player in 2024
    prev_posts = posteriors_by_season[2023]
    prior_dict = build_sequential_priors(prev_posts, [thin_player.player_id], hyperparams, lam=1.25)
    prior = prior_dict[thin_player.player_id]

    fig9 = plot_prior_posterior_update(
        thin_player,
        prior_mu_a=prior["prior_mu_a"],
        prior_sigma_a=prior["prior_sigma_a"],
        prior_mu_b=prior["prior_mu_b"],
        prior_sigma_b=prior["prior_sigma_b"],
    )
    fig9.show()
    fig9.write_html("../docs/bayes_prior_posterior.html", include_plotlyjs="cdn")
    print(f"Saved bayes_prior_posterior.html ({thin_player.player_name}, n={thin_player.n_opportunities})")
except FileNotFoundError:
    print(f"Burn-in hyperparams not yet cached at {hp_path}. Run Phase 0 first.")

# %%
# ── Within-season evolution (2024) ────────────────────────────────────────────
# Fit monthly snapshots through 2024 season. Slower — uses n_samples=500.
# Comment out if you don't need the evolution figure.

from baseball_range.bayes import fit_season_evolution
from baseball_range.viz import plot_season_evolution

print("Fitting 2024 season evolution (monthly snapshots)...")
evolution = fit_season_evolution(
    df_season=data_by_season[2024],
    season=2024,
    initial_posteriors=posteriors_by_season[2023],
    hyperparams=hyperparams,
    stan_update=str(STAN_DIR / "cf_range_update_v2.stan"),
    lam=DEFAULT_LAM_SCHEDULE.get(2024, 1.25),
    n_chains=2,
    n_samples=500,
    seed=0,
)

fig8 = plot_season_evolution(evolution, player_ids=top_ids[:4])
fig8.show()
fig8.write_html("../docs/bayes_season_evolution.html", include_plotlyjs="cdn")
print("Saved bayes_season_evolution.html")

# %%
# ── λ sensitivity ─────────────────────────────────────────────────────────────
# How much do rankings change as λ varies? Stable rankings indicate the result
# is robust to the amount of year-to-year prior inflation.
#
# Run on 2024 only; compare rank order across λ ∈ {1.05, 1.25, 1.5, 2.0}.

from baseball_range.bayes import fit_season

lam_grid = [1.05, 1.25, 1.5, 2.0]
rank_tables = {}
prev_posts_2023 = posteriors_by_season[2023]

for lam in lam_grid:
    posts = fit_season(
        data_by_season[2024], 2024,
        prev_posteriors=prev_posts_2023,
        hyperparams=hyperparams,
        stan_file=str(STAN_DIR / "cf_range_update_v2.stan"),
        lam=lam,
        n_chains=4, n_samples=1000, seed=42,
    )
    stats = bayes_compute_all_stats(posts, df_analysis)
    rank_tables[lam] = stats["player_name"].tolist()

# Show rank differences
base_ranks = {name: i for i, name in enumerate(rank_tables[1.25])}
print("\nλ sensitivity: rank shift from λ=1.25 baseline (top-15 players)")
print(f"{'Player':<26s}  {'λ=1.05':>7s}  {'λ=1.50':>7s}  {'λ=2.00':>7s}")
for name in rank_tables[1.25][:15]:
    shifts = [
        rank_tables[lam].index(name) - base_ranks[name]
        for lam in [1.05, 1.50, 2.00]
    ]
    print(f"  {name:<26s}  {shifts[0]:+5d}    {shifts[1]:+5d}    {shifts[2]:+5d}")
