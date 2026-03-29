# %% [markdown]
# # 07 — v2.0 Validation
#
# Two validations:
#
# **A. OAA correlation** — does OWR/spectacular_play_prob agree with Statcast OAA?
#   - A1: OWR vs season OAA (scatter, Spearman ρ by year)
#   - A2: gamma_mean and spectacular_play_prob vs directional OAA (In vs Back zones)
#   - A3: Shrinkage demonstration — Bayesian estimates closer to OAA for thin-data players
#
# **B. y0_offset face validity** — does the inferred starting depth make sense?
#   - B1: Population distribution (expected mean ~+10–20 ft)
#   - B2: Top/bottom player ranking by y0_offset — smell test
#   - B3: of_fielding_alignment residual check (Standard vs Strategic alignment)
#
# Note: direct coordinate validation (actual fielder x/y at pitch time) is not
# available in the public pybaseball API. Hawk-Eye position tracking is not released.
# Validations B1/B2 are face-validity checks; B3 is an indirect consistency test.

# %%
import sys
sys.path.insert(0, "../src")

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats as scipy_stats
from pybaseball import statcast_outs_above_average, statcast_outfield_directional_oaa

pio.renderers.default = "browser"

from baseball_range.bayes import load_posteriors, bayes_per_play_predictions
from baseball_range.data import add_player_names, pull_cf_opportunities

CACHE_DIR = pathlib.Path("../data")
DOCS_DIR  = pathlib.Path("../docs")

UPDATE_SEASONS = [2021, 2022, 2023, 2024]


# %%
# ── Load v2.0 stats and posteriors ────────────────────────────────────────────

print("Loading v2.0 stats and posteriors...")
stats_by_season = {s: pd.read_parquet(CACHE_DIR / f"bayes_stats_{s}.parquet")
                   for s in UPDATE_SEASONS}
posteriors_by_season = {s: load_posteriors(str(CACHE_DIR / f"posteriors_{s}"))
                        for s in UPDATE_SEASONS}

for s in UPDATE_SEASONS:
    print(f"  {s}: {len(stats_by_season[s])} players")

# %%
# ── Pull OAA from Baseball Savant ─────────────────────────────────────────────
# pos=8 is center field. min_att=20 keeps thin-data players for shrinkage demo.
# Column names may vary by pybaseball version — inspect and adjust below.

print("\nPulling OAA data (2021–2024)...")
oaa_by_season = {}
for season in UPDATE_SEASONS:
    cache_path = CACHE_DIR / f"oaa_cf_{season}.parquet"
    if cache_path.exists():
        oaa_by_season[season] = pd.read_parquet(cache_path)
        print(f"  {season}: loaded from cache")
    else:
        df = statcast_outs_above_average(season, pos=8, min_att=20)
        df.to_parquet(cache_path, index=False)
        oaa_by_season[season] = df
        print(f"  {season}: pulled {len(df)} players, cached")

# Inspect columns on first season to find the player ID and OAA columns
print("\nOAA columns:", oaa_by_season[UPDATE_SEASONS[0]].columns.tolist())
print(oaa_by_season[UPDATE_SEASONS[0]].head(3).to_string())

# %%
# Directional OAA
print("\nPulling directional OAA (2021–2024)...")
dir_oaa_by_season = {}
for season in UPDATE_SEASONS:
    cache_path = CACHE_DIR / f"oaa_directional_{season}.parquet"
    if cache_path.exists():
        dir_oaa_by_season[season] = pd.read_parquet(cache_path)
        print(f"  {season}: loaded from cache")
    else:
        df = statcast_outfield_directional_oaa(season, min_opp=20)
        df.to_parquet(cache_path, index=False)
        dir_oaa_by_season[season] = df
        print(f"  {season}: pulled {len(df)} players, cached")

print("\nDirectional OAA columns:", dir_oaa_by_season[UPDATE_SEASONS[0]].columns.tolist())
print(dir_oaa_by_season[UPDATE_SEASONS[0]].head(3).to_string())

# %%
# ── Normalize OAA column names ────────────────────────────────────────────────
# pybaseball column names vary. Detect and standardize.

def find_col(df, candidates):
    """Return first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

sample_oaa = oaa_by_season[UPDATE_SEASONS[0]]
OAA_ID_COL  = find_col(sample_oaa, ["player_id", "mlb_id", "mlbam_id", "id"])
OAA_OAA_COL = find_col(sample_oaa, ["outs_above_average", "oaa", "OAA"])
OAA_N_COL   = next((c for c in ["n", "attempts", "n_outs_above_average", "total_bolts"]
                    if c in sample_oaa.columns), None)
print(f"\nOAA columns: id={OAA_ID_COL}, oaa={OAA_OAA_COL}, n={OAA_N_COL}")

sample_dir = dir_oaa_by_season[UPDATE_SEASONS[0]]
DIR_ID_COL  = find_col(sample_dir, ["player_id", "mlb_id", "mlbam_id", "id"])
# Directional OAA columns — look for "in" and "back" in column names
in_cols   = [c for c in sample_dir.columns if "in"   in c.lower() and "oaa" in c.lower()]
back_cols = [c for c in sample_dir.columns if "back" in c.lower() and "oaa" in c.lower()]
print(f"Directional OAA 'In' cols: {in_cols}")
print(f"Directional OAA 'Back' cols: {back_cols}")


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# A1: OWR vs season OAA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A1: OWR vs season OAA")
print("="*70)

merged_by_season = {}
rho_by_season = {}

for season in UPDATE_SEASONS:
    oaa_cols = [OAA_ID_COL, OAA_OAA_COL] + ([OAA_N_COL] if OAA_N_COL else [])
    rename_map = {OAA_ID_COL: "player_id", OAA_OAA_COL: "oaa"}
    if OAA_N_COL:
        rename_map[OAA_N_COL] = "oaa_n"
    oaa = oaa_by_season[season][oaa_cols].rename(columns=rename_map)
    merged = stats_by_season[season].merge(oaa, on="player_id", how="inner")
    rho, pval = scipy_stats.spearmanr(merged["opp_weighted_range"], merged["oaa"])
    merged_by_season[season] = merged
    rho_by_season[season] = (rho, pval, len(merged))
    print(f"  {season}: n={len(merged)}, Spearman ρ(OWR, OAA) = {rho:+.3f}  (p={pval:.4f})")

print("\nKey story: does ρ improve from 2021→2024 as the prior chain accumulates?")
rhos = [rho_by_season[s][0] for s in UPDATE_SEASONS]
if rhos[-1] > rhos[0] + 0.05:
    print("  → YES — Bayesian pipeline earns its keep as evidence accumulates")
elif rhos[-1] < rhos[0] - 0.05:
    print("  → NO — correlation weakens (investigate)")
else:
    print("  → STABLE across seasons")

# Figure: 4-panel scatter (one per season)
fig_a1 = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{s} (ρ={rho_by_season[s][0]:+.3f})"
                                        for s in UPDATE_SEASONS])
for i, season in enumerate(UPDATE_SEASONS):
    row, col = divmod(i, 2)
    m = merged_by_season[season]
    top3  = m.nlargest(3, "opp_weighted_range")
    bot3  = m.nsmallest(3, "opp_weighted_range")
    labeled = pd.concat([top3, bot3])
    other   = m[~m["player_id"].isin(labeled["player_id"])]
    fig_a1.add_trace(go.Scatter(
        x=other["opp_weighted_range"], y=other["oaa"],
        mode="markers",
        marker=dict(size=np.sqrt(other["n_opportunities"].clip(1)) * 1.5,
                    color="#aec7e8", opacity=0.7),
        hovertemplate="<b>%{text}</b><br>OWR=%{x:.3f}<br>OAA=%{y}<extra></extra>",
        text=other["player_name"],
        showlegend=False,
    ), row=row+1, col=col+1)
    fig_a1.add_trace(go.Scatter(
        x=labeled["opp_weighted_range"], y=labeled["oaa"],
        mode="markers+text",
        marker=dict(size=np.sqrt(labeled["n_opportunities"].clip(1)) * 1.5,
                    color="#1f77b4"),
        text=labeled["player_name"].str.split().str[-1],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>OWR=%{x:.3f}<br>OAA=%{y}<extra></extra>",
        showlegend=False,
    ), row=row+1, col=col+1)

fig_a1.update_xaxes(title_text="OWR")
fig_a1.update_yaxes(title_text="OAA")
fig_a1.update_layout(
    title="A1: Opportunity-Weighted Range vs Statcast OAA (CF, 2021–2024)",
    width=900, height=780,
)
fig_a1.show()
fig_a1.write_html(str(DOCS_DIR / "val_owr_vs_oaa.html"), include_plotlyjs="cdn")
print("Saved val_owr_vs_oaa.html")


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# A2: gamma and spectacular_play_prob vs directional OAA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A2: gamma / spectacular_play_prob vs directional OAA")
print("="*70)

if in_cols and back_cols:
    in_col   = in_cols[0]
    back_col = back_cols[0]
    print(f"Using: in={in_col}, back={back_col}")

    dir_merged_seasons = []
    for season in UPDATE_SEASONS:
        d = dir_oaa_by_season[season][[DIR_ID_COL, in_col, back_col]].rename(
            columns={DIR_ID_COL: "player_id",
                     in_col: "oaa_in", back_col: "oaa_back"}
        )
        # "In" fraction: share of directional value from charging plays
        # Use absolute values to handle negative OAA players
        d["in_frac"] = d["oaa_in"] / (d["oaa_in"].abs() + d["oaa_back"].abs() + 1e-9)
        m = stats_by_season[season].merge(d, on="player_id", how="inner")
        m["season"] = season
        dir_merged_seasons.append(m)
    dir_merged = pd.concat(dir_merged_seasons, ignore_index=True)

    # Correlate gamma_mean vs in_frac
    rho_gam, p_gam = scipy_stats.spearmanr(dir_merged["gamma_mean"].dropna(),
                                             dir_merged.loc[dir_merged["gamma_mean"].notna(), "in_frac"])
    rho_spec, p_spec = scipy_stats.spearmanr(dir_merged["spectacular_play_prob_mean"],
                                              dir_merged["oaa_in"])
    print(f"  Spearman ρ(gamma_mean, OAA In-fraction) = {rho_gam:+.3f}  (p={p_gam:.4f})")
    print(f"  Spearman ρ(spectacular_play_prob, OAA In) = {rho_spec:+.3f}  (p={p_spec:.4f})")

    if rho_gam > 0.3 and p_gam < 0.05:
        print("  → gamma correlates with In-zone OAA: charge asymmetry identified is real")
    elif abs(rho_gam) < 0.15:
        print("  → gamma does not predict directional OAA: may be prior-dominated or OAA insufficient")

    # Figure: 2-panel scatter
    fig_a2 = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                f"gamma vs OAA In-fraction (ρ={rho_gam:+.3f})",
                                f"Spectacular vs OAA In (ρ={rho_spec:+.3f})",
                            ])
    fig_a2.add_trace(go.Scatter(
        x=dir_merged["gamma_mean"], y=dir_merged["in_frac"],
        mode="markers",
        marker=dict(size=6, color="#2ca02c", opacity=0.6),
        text=dir_merged["player_name"],
        hovertemplate="<b>%{text}</b><br>γ=%{x:.3f}<br>In-frac=%{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig_a2.add_trace(go.Scatter(
        x=dir_merged["spectacular_play_prob_mean"], y=dir_merged["oaa_in"],
        mode="markers",
        marker=dict(size=6, color="#d62728", opacity=0.6),
        text=dir_merged["player_name"],
        hovertemplate="<b>%{text}</b><br>spec=%{x:.3f}<br>OAA In=%{y}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig_a2.update_xaxes(title_text="gamma mean", row=1, col=1)
    fig_a2.update_yaxes(title_text="OAA In-fraction", row=1, col=1)
    fig_a2.update_xaxes(title_text="spectacular_play_prob", row=1, col=2)
    fig_a2.update_yaxes(title_text="OAA In (charging)", row=1, col=2)
    fig_a2.update_layout(
        title="A2: Charge asymmetry and spectacular plays vs directional OAA (2021–2024)",
        width=880, height=440,
    )
    fig_a2.show()
    fig_a2.write_html(str(DOCS_DIR / "val_spectacular_vs_oaa.html"), include_plotlyjs="cdn")
    print("Saved val_spectacular_vs_oaa.html")
else:
    print("  Could not identify In/Back OAA column names — skipping A2.")
    print(f"  Available columns: {sample_dir.columns.tolist()}")


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# A3: Shrinkage demonstration
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A3: Shrinkage demonstration — Bayesian closer to OAA than MLE for thin-data players")
print("="*70)

# Use 2024; compare Bayesian OWR rank vs OAA rank for all merged players
season = 2024
m = merged_by_season[season].copy()
m = m.sort_values("opp_weighted_range", ascending=False).reset_index(drop=True)
m["bayes_rank"] = m.index + 1
m = m.sort_values("oaa", ascending=False).reset_index(drop=True)
m["oaa_rank"] = m.index + 1
m["rank_gap"] = (m["bayes_rank"] - m["oaa_rank"]).abs()

# MLE comparison: use posterior means as the "Bayes" estimate and posterior means
# without shrinkage as "MLE proxy". Shrinkage is implicit in the posterior — as a
# proxy for MLE, use a_mean * b_mean as an unscaled area estimate and rank by that.
m["mle_proxy"] = m["a_mean"] * m["b_mean"]
m_sorted_mle = m.sort_values("mle_proxy", ascending=False).reset_index(drop=True)
m["mle_rank"] = m_sorted_mle.index + 1
m["mle_rank_gap"] = (m["mle_rank"] - m["oaa_rank"]).abs()

print(f"\n2024: n={len(m)} merged players")
print(f"  Mean |Bayes rank − OAA rank| = {m['rank_gap'].mean():.1f}")
print(f"  Mean |MLE proxy rank − OAA rank| = {m['mle_rank_gap'].mean():.1f}")

# For thin-data players specifically
thin = m[m["n_opportunities"] < 75]
thick = m[m["n_opportunities"] >= 75]
if len(thin) > 3:
    print(f"\n  Thin-data players (<75 opps, n={len(thin)}):")
    print(f"    |Bayes rank − OAA rank| mean = {thin['rank_gap'].mean():.1f}")
    print(f"    |MLE proxy rank − OAA rank| mean = {thin['mle_rank_gap'].mean():.1f}")
if len(thick) > 3:
    print(f"\n  High-data players (≥75 opps, n={len(thick)}):")
    print(f"    |Bayes rank − OAA rank| mean = {thick['rank_gap'].mean():.1f}")
    print(f"    |MLE proxy rank − OAA rank| mean = {thick['mle_rank_gap'].mean():.1f}")

print("\n  (Shrinkage should help thin-data players; high-data players should be similar)")

# Scatter: rank gap vs n_opportunities
fig_a3 = go.Figure()
fig_a3.add_trace(go.Scatter(
    x=m["n_opportunities"], y=m["rank_gap"],
    mode="markers", name="Bayesian OWR",
    marker=dict(size=7, color="#1f77b4", opacity=0.7),
    text=m["player_name"],
    hovertemplate="<b>%{text}</b><br>n=%{x}<br>|rank gap|=%{y}<extra></extra>",
))
fig_a3.add_trace(go.Scatter(
    x=m["n_opportunities"], y=m["mle_rank_gap"],
    mode="markers", name="MLE proxy",
    marker=dict(size=7, color="#d62728", opacity=0.5, symbol="x"),
    text=m["player_name"],
    hovertemplate="<b>%{text}</b><br>n=%{x}<br>|rank gap|=%{y}<extra></extra>",
))
fig_a3.add_vline(x=75, line_dash="dot", line_color="gray",
                  annotation_text="75 opps threshold", annotation_position="top right")
fig_a3.update_layout(
    title="A3: Rank gap vs OAA — Bayesian shrinkage benefit for thin-data players (2024)",
    xaxis_title="n_opportunities",
    yaxis_title="|Bayes/MLE rank − OAA rank|",
    width=700, height=440,
)
fig_a3.show()


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# B1: Population distribution of y0_offset
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B1: y0_offset population distribution")
print("="*70)

y0_all = []
for season in UPDATE_SEASONS:
    for bpr in posteriors_by_season[season]:
        if bpr.y0_offset_samples is not None:
            y0_all.append(bpr.y0_offset_mean)
y0_all = np.array(y0_all)

print(f"All players, all seasons (n={len(y0_all)}):")
print(f"  mean  = {y0_all.mean():+.2f} ft  (expected ~+10–20 ft)")
print(f"  median= {np.median(y0_all):+.2f} ft")
print(f"  SD    = {y0_all.std():.2f} ft")
print(f"  5th–95th pct: {np.percentile(y0_all,5):+.1f} to {np.percentile(y0_all,95):+.1f} ft")

if y0_all.mean() < 5:
    print("  → mean < 5 ft: y0_offset may be underestimated (check prior)")
elif y0_all.mean() > 30:
    print("  → mean > 30 ft: unexpectedly deep positioning (investigate)")
else:
    print("  → Population mean in expected range ✓")

# Normal fit overlay
mu_fit, sd_fit = y0_all.mean(), y0_all.std()
x_fit = np.linspace(y0_all.min() - 5, y0_all.max() + 5, 200)
y_fit = scipy_stats.norm.pdf(x_fit, mu_fit, sd_fit) * len(y0_all) * (y0_all.max()-y0_all.min()) / 30

fig_b1 = go.Figure()
fig_b1.add_trace(go.Histogram(x=y0_all, nbinsx=30, name="observed",
                               marker_color="#aec7e8", opacity=0.8))
fig_b1.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
                              line=dict(color="#1f77b4", width=2), name="normal fit"))
fig_b1.add_vline(x=0, line_dash="dot", line_color="gray",
                  annotation_text="canonical 310 ft", annotation_position="top right")
fig_b1.update_layout(
    title=f"B1: y0_offset distribution (all players, 2021–2024)<br>"
          f"mean={mu_fit:+.1f} ft, SD={sd_fit:.1f} ft",
    xaxis_title="y0_offset_mean (ft from canonical 310 ft)",
    yaxis_title="Count",
    width=600, height=380,
)
fig_b1.show()


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# B2: Face-validity ranking
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B2: Face-validity ranking of y0_offset (2024)")
print("="*70)
print("NOTE: This is a qualitative smell test, not a formal validation.")
print("Players known to play shallow should rank low; deep-playing CFs high.\n")

y0_rows = []
for bpr in posteriors_by_season[2024]:
    if bpr.y0_offset_samples is not None:
        y0_rows.append({
            "player_name":    bpr.player_name,
            "y0_offset_mean": bpr.y0_offset_mean,
            "y0_offset_sd":   bpr.y0_offset_sd,
            "n_opportunities": bpr.n_opportunities,
            "approx_depth_ft": 310 + bpr.y0_offset_mean,
        })
y0_df = pd.DataFrame(y0_rows).sort_values("y0_offset_mean", ascending=False)

print("Top 10 — deepest estimated starting position:")
print(y0_df.head(10)[["player_name","approx_depth_ft","y0_offset_mean","y0_offset_sd","n_opportunities"]]
      .to_string(index=False, float_format="%.1f"))

print("\nBottom 10 — shallowest estimated starting position:")
print(y0_df.tail(10)[["player_name","approx_depth_ft","y0_offset_mean","y0_offset_sd","n_opportunities"]]
      .to_string(index=False, float_format="%.1f"))


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# B3: of_fielding_alignment residual check (2024)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B3: of_fielding_alignment residual check (2024)")
print("="*70)
print("Hypothesis: 'Strategic' alignment plays should show larger residuals")
print("because our fixed y0_offset assumption is weakest for atypical alignments.")
print()

ALIGNED_CACHE = CACHE_DIR / "cf_2024_aligned.parquet"

if ALIGNED_CACHE.exists():
    print("Loading from cache...")
    df_aligned = pd.read_parquet(ALIGNED_CACHE)
else:
    print("Pulling 2024 Statcast with of_fielding_alignment (~15 min)...")
    from pybaseball import statcast
    raw = statcast(start_dt="2024-04-01", end_dt="2024-10-01")
    flies = raw[raw["bb_type"] == "fly_ball"].copy()
    required = ["hc_x", "hc_y", "launch_speed", "launch_angle", "fielder_8", "events"]
    flies = flies.dropna(subset=required)

    from baseball_range.data import compute_hang_time, pixels_to_feet, CF_LAT_MAX, CF_DEPTH_MIN, CF_DEPTH_MAX, CF_X0, CF_Y0
    flies["hang_time"] = compute_hang_time(flies["launch_speed"], flies["launch_angle"])
    flies["feet_x"], flies["feet_y"] = pixels_to_feet(flies["hc_x"], flies["hc_y"])
    cf_mask = (
        (flies["feet_x"].abs() < CF_LAT_MAX) &
        (flies["feet_y"] > CF_DEPTH_MIN) &
        (flies["feet_y"] < CF_DEPTH_MAX) &
        (flies["hang_time"] > 0)
    )
    cf = flies[cf_mask].copy()
    cf["delta_x"] = cf["feet_x"] - CF_X0
    cf["delta_y"]  = cf["feet_y"] - CF_Y0
    cf["caught"]   = cf["events"].isin(["field_out", "sac_fly"]).astype(int)
    cf["player_id"] = cf["fielder_8"].astype(int)
    keep = ["game_date", "player_id", "delta_x", "delta_y", "hang_time", "caught",
            "feet_x", "feet_y", "of_fielding_alignment"]
    df_aligned = cf[[c for c in keep if c in cf.columns]].reset_index(drop=True)
    df_aligned.to_parquet(ALIGNED_CACHE, index=False)
    print(f"  Pulled {len(df_aligned):,} plays, cached to cf_2024_aligned.parquet")

if "of_fielding_alignment" not in df_aligned.columns:
    print("  of_fielding_alignment not available in this data — skipping B3")
else:
    print(f"  alignment value counts:\n{df_aligned['of_fielding_alignment'].value_counts().to_string()}")

    # Compute v2.0 residuals (reuse 2024 posteriors against aligned data)
    posts_map = {bpr.player_id: bpr for bpr in posteriors_by_season[2024]}
    preds_rows = []
    for pid, grp in df_aligned.groupby("player_id"):
        bpr = posts_map.get(pid)
        if bpr is None:
            continue
        p = bayes_per_play_predictions(bpr, grp)
        if len(p):
            p["of_fielding_alignment"] = grp["of_fielding_alignment"].values[:len(p)]
            preds_rows.append(p)
    if preds_rows:
        preds_aligned = pd.concat(preds_rows, ignore_index=True)
        # Normalize alignment values
        preds_aligned["alignment"] = (
            preds_aligned["of_fielding_alignment"]
            .fillna("Standard")
            .str.strip()
        )
        align_stats = (
            preds_aligned.groupby("alignment")["residual"]
            .agg(mean="mean", rmse=lambda x: np.sqrt((x**2).mean()), n="count")
            .reset_index()
        )
        print("\n  Residuals by of_fielding_alignment:")
        print(align_stats.to_string(index=False, float_format="%.4f"))

        standard_rmse  = align_stats.loc[align_stats["alignment"] == "Standard",  "rmse"].values
        strategic_rmse = align_stats.loc[align_stats["alignment"].str.contains("Strategic", case=False), "rmse"].values
        if len(standard_rmse) and len(strategic_rmse):
            if strategic_rmse[0] > standard_rmse[0] + 0.01:
                print("  → Strategic alignment plays show higher RMSE ✓ (consistent with hypothesis)")
            else:
                print("  → No material RMSE difference between alignment types")
    else:
        print("  No predictions computed — player IDs may not overlap")


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

print("\nA. OAA correlation:")
for season in UPDATE_SEASONS:
    rho, pval, n = rho_by_season[season]
    print(f"  {season}: ρ(OWR, OAA) = {rho:+.3f}  (p={pval:.4f}, n={n})")

if in_cols and back_cols:
    print(f"\n  ρ(gamma, OAA In-fraction) = {rho_gam:+.3f}  (p={p_gam:.4f})")
    print(f"  ρ(spectacular_prob, OAA In) = {rho_spec:+.3f}  (p={p_spec:.4f})")

print(f"\nB. y0_offset:")
print(f"  Population mean = {y0_all.mean():+.2f} ft  (expected ~+10–20 ft)")
print(f"  See B2 table for face-validity ranking")
print(f"  B3 alignment check: see residual table above")

print("\nFigures saved:")
print("  val_owr_vs_oaa.html          A1 — OWR vs Statcast OAA")
print("  val_spectacular_vs_oaa.html  A2 — directional OAA validates gamma")
