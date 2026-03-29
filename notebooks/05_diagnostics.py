# %% [markdown]
# # 05 — Ellipse Diagnostic Battery
#
# Validates the parametric ellipse geometry against the 2021-2024 data.
# Runs five diagnostics in priority order:
#
# 1. **Spatial residuals** — per-play scatter on (Δx, Δy) field grid
# 2. **Directional decomposition** — mean residual by 8×45° approach-angle bins
# 3. **Calibration** — actual catch rate vs. predicted P(catch) across all plays
# 4. **Shape-ratio vs RMSE** — do extreme-shape players fit worse?
# 5. **Moran's I** — formal spatial autocorrelation test per high-power player
#
# ## Decision gate
# | Result | Action |
# |---|---|
# | Clean residuals, calibration on diagonal, Moran's I not significant | Ellipse confirmed — proceed to Part 2 |
# | Systematic spatial clustering, significant Moran's I | Fit superellipse Stan model |
# | Calibration S-curve in mid-range | Consider semiparametric decay |

# %%
import sys
sys.path.insert(0, "../src")

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

from baseball_range.bayes import (
    load_posteriors, bayes_per_play_predictions,
)
from baseball_range.viz import plot_spatial_residuals, plot_calibration

CACHE_DIR = pathlib.Path("../data")
DOCS_DIR  = pathlib.Path("../docs")

# %%
# ── Load posteriors and data ──────────────────────────────────────────────────

UPDATE_SEASONS = [2021, 2022, 2023, 2024]

posteriors_by_season = {}
for season in UPDATE_SEASONS:
    posteriors_by_season[season] = load_posteriors(
        str(CACHE_DIR / f"posteriors_{season}")
    )

data_by_season = {}
for season in UPDATE_SEASONS:
    df = pd.read_parquet(CACHE_DIR / f"cf_{season}.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    data_by_season[season] = df

stats_by_season = {s: pd.read_parquet(CACHE_DIR / f"bayes_stats_{s}.parquet")
                   for s in UPDATE_SEASONS}

# Combined analysis dataset (4 seasons, same as stats computation)
df_analysis = pd.concat(list(data_by_season.values()), ignore_index=True)

print(f"Analysis dataset: {len(df_analysis):,} plays across {len(UPDATE_SEASONS)} seasons")

# %%
# ── Compute per-play predictions for all players (2024 posteriors) ───────────
# Using 2024 posteriors and their own season's data.
# For cross-season diagnostics, loop all seasons below.

print("Computing per-play predictions (2024)...")
preds_2024 = []
for bpr in posteriors_by_season[2024]:
    p = bayes_per_play_predictions(bpr, data_by_season[2024])
    if len(p):
        p["player_name"] = bpr.player_name
        p["player_id"] = bpr.player_id
        preds_2024.append(p)
preds_2024 = pd.concat(preds_2024, ignore_index=True)
print(f"  {len(preds_2024):,} plays, {preds_2024['player_id'].nunique()} players")

# %%
# ── Diagnostic 1: Spatial residual plots ─────────────────────────────────────
# Plot for every player with ≥150 opps in 2024.

MIN_OPPS_SPATIAL = 150
posts_map = {bpr.player_id: bpr for bpr in posteriors_by_season[2024]}
high_power = (
    stats_by_season[2024]
    .query("n_opportunities >= @MIN_OPPS_SPATIAL")
    .sort_values("opp_weighted_range", ascending=False)
)
print(f"\nSpatial residual plots for {len(high_power)} players with ≥{MIN_OPPS_SPATIAL} opps (2024):")

spatial_figs = {}
for _, row in high_power.iterrows():
    pid = row["player_id"]
    bpr = posts_map[pid]
    p = preds_2024[preds_2024["player_id"] == pid]
    fig = plot_spatial_residuals(bpr, p)
    fig.show()
    spatial_figs[bpr.player_name] = fig
    print(f"  {bpr.player_name}: n={len(p)}, mean_resid={p['residual'].mean():+.4f}, "
          f"RMSE={np.sqrt((p['residual']**2).mean()):.4f}")

# %%
# ── Diagnostic 2: Directional decomposition ──────────────────────────────────
# 8 × 45° bins of approach angle θ = atan2(Δy, Δx).
# Each bar = mean residual ± SE for that sector.
# A well-specified ellipse should have near-zero residuals in all sectors.

preds_2024["theta_deg"] = np.degrees(
    np.arctan2(preds_2024["delta_y"], preds_2024["delta_x"])
)
# Shift from (-180,180] to [0,360)
preds_2024["theta_deg"] = preds_2024["theta_deg"] % 360

BIN_EDGES = np.arange(0, 361, 45)
BIN_LABELS = [f"{int(lo)}–{int(lo+45)}°" for lo in BIN_EDGES[:-1]]
preds_2024["sector"] = pd.cut(
    preds_2024["theta_deg"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
)

# Pooled across all players — systematic pattern across players is the signal
sector_stats = (
    preds_2024.groupby("sector", observed=True)["residual"]
    .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)), n="count")
    .reset_index()
)

fig_dir = go.Figure()
fig_dir.add_trace(go.Bar(
    x=sector_stats["sector"].astype(str),
    y=sector_stats["mean"],
    error_y=dict(type="data", array=sector_stats["se"], visible=True),
    marker_color=[
        "#d6604d" if v > 0 else "#2166ac" for v in sector_stats["mean"]
    ],
    text=[f"n={int(r.n)}" for r in sector_stats.itertuples()],
    hovertemplate="%{x}<br>mean resid=%{y:.4f}<br>%{text}<extra></extra>",
))
fig_dir.add_hline(y=0, line_dash="dot", line_color="gray")
fig_dir.update_layout(
    title="Directional decomposition — mean residual by approach angle (2024, all players)",
    xaxis_title="Approach angle θ = atan2(Δy, Δx)",
    yaxis_title="Mean residual (caught − p̂)",
    width=700, height=420,
)
fig_dir.show()
print("\nDirectional residuals (2024, pooled):")
print(sector_stats[["sector", "mean", "se", "n"]].to_string(index=False, float_format="%.4f"))

# Per-player directional decomposition for high-power players
print("\nPer-player sector means (rows=player, cols=sector, values=mean residual):")
pivot_rows = []
for pid, grp in preds_2024.groupby("player_id"):
    bpr = posts_map.get(pid)
    if bpr is None or len(grp) < MIN_OPPS_SPATIAL:
        continue
    sec = grp.groupby("sector", observed=True)["residual"].mean()
    row = {"player": bpr.player_name}
    row.update(sec.to_dict())
    pivot_rows.append(row)
if pivot_rows:
    pivot = pd.DataFrame(pivot_rows).set_index("player")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

# %%
# ── Diagnostic 3: Calibration plot ───────────────────────────────────────────
# Pool all 2024 plays; bin by predicted P(catch).

fig_cal = plot_calibration(preds_2024, n_bins=10)
fig_cal.show()
fig_cal.write_html(str(DOCS_DIR / "diag_calibration.html"), include_plotlyjs="cdn")
print("\nSaved diag_calibration.html")

# Also show calibration broken down by distance bucket (in/out/on-boundary)
preds_2024["d_bucket"] = pd.cut(
    preds_2024["p_catch_mean"],
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=["hard (<0.2)", "mid-low (0.2-0.5)", "mid-high (0.5-0.8)", "easy (>0.8)"],
)
bucket_stats = (
    preds_2024.groupby("d_bucket", observed=True)
    .agg(p_mean=("p_catch_mean", "mean"), catch_rate=("caught", "mean"), n=("caught", "count"))
    .reset_index()
)
print("\nCalibration by catch-probability bucket (2024):")
print(bucket_stats.to_string(index=False, float_format="%.3f"))

# %%
# ── Diagnostic 4: Shape-ratio vs residual RMSE ───────────────────────────────
# Does having an extreme a/b ratio predict worse fit?

rmse_rows = []
for _, stat_row in stats_by_season[2024].iterrows():
    if stat_row["n_opportunities"] < 75:
        continue
    pid = stat_row["player_id"]
    p = preds_2024[preds_2024["player_id"] == pid]
    if len(p) < 20:
        continue
    rmse = np.sqrt((p["residual"] ** 2).mean())
    shape_ratio = stat_row["a_mean"] / stat_row["b_mean"]
    rmse_rows.append({
        "player_name": stat_row["player_name"],
        "shape_ratio": shape_ratio,
        "rmse": rmse,
        "n": len(p),
    })
rmse_df = pd.DataFrame(rmse_rows)

from scipy import stats as scipy_stats
median_ratio = rmse_df["shape_ratio"].median()
rmse_df["shape_ratio_dev"] = (rmse_df["shape_ratio"] - median_ratio).abs()
rho, pval = scipy_stats.spearmanr(rmse_df["shape_ratio_dev"], rmse_df["rmse"])

fig_shape = go.Figure(go.Scatter(
    x=rmse_df["shape_ratio_dev"],
    y=rmse_df["rmse"],
    mode="markers",
    marker=dict(size=np.sqrt(rmse_df["n"]) * 1.5, color="#1f77b4", opacity=0.7),
    text=rmse_df["player_name"],
    hovertemplate="<b>%{text}</b><br>|a/b − median|=%{x:.2f}<br>RMSE=%{y:.4f}<extra></extra>",
))
fig_shape.update_layout(
    title=f"Shape-ratio deviation vs. residual RMSE (2024, ≥75 opps)<br>"
          f"Spearman ρ={rho:.3f}, p={pval:.3f}",
    xaxis_title="|a/b − median(a/b)|",
    yaxis_title="Residual RMSE",
    width=560, height=480,
)
fig_shape.show()
print(f"\nShape-ratio vs RMSE: Spearman ρ={rho:.3f}, p={pval:.4f}")
print(rmse_df.sort_values("shape_ratio_dev", ascending=False).head(10)
      .to_string(index=False, float_format="%.3f"))

# %%
# ── Diagnostic 5: Moran's I spatial autocorrelation ──────────────────────────
# Formal test: are residuals spatially clustered for high-power players?
# H0: spatial randomness. Significant positive I → ellipse wrong for that player.
# Uses permutation test (1000 shuffles) — no external spatial stats package needed.

from scipy.spatial import distance_matrix

def morans_i(coords, z, threshold=30.0, n_perm=1000, rng=None):
    """
    Moran's I with permutation test.

    coords : (N, 2) array of (x, y) positions
    z      : (N,) residuals
    threshold : spatial weight cutoff in ft
    Returns: (I, p_value, z_score)
    """
    rng = np.random.default_rng(rng)
    D = distance_matrix(coords, coords)
    W = (D < threshold).astype(float)
    np.fill_diagonal(W, 0)
    W_sum = W.sum()
    if W_sum == 0:
        return np.nan, np.nan, np.nan
    N = len(z)
    z_c = z - z.mean()
    I_obs = (N / W_sum) * (z_c @ W @ z_c) / (z_c @ z_c)
    # Permutation null
    I_perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = rng.permutation(z_c)
        I_perm[k] = (N / W_sum) * (zp @ W @ zp) / (z_c @ z_c)
    p_val = (I_perm >= I_obs).mean()
    z_score = (I_obs - I_perm.mean()) / I_perm.std()
    return I_obs, p_val, z_score


MORAN_THRESHOLD = 30  # ft — plays within 30 ft are "neighbors"
MORAN_NPERM     = 1000

print(f"\nMoran's I spatial autocorrelation (threshold={MORAN_THRESHOLD} ft, "
      f"{MORAN_NPERM} permutations)")
print(f"{'Player':<24s}  {'n':>5s}  {'I':>7s}  {'z':>7s}  {'p':>7s}  {'flag':>6s}")
print("─" * 65)

moran_rows = []
for _, stat_row in (
    stats_by_season[2024]
    .query("n_opportunities >= 150")
    .sort_values("opp_weighted_range", ascending=False)
    .iterrows()
):
    pid = stat_row["player_id"]
    p = preds_2024[preds_2024["player_id"] == pid]
    if len(p) < 50:
        continue
    coords = p[["delta_x", "delta_y"]].to_numpy()
    z = p["residual"].to_numpy()
    I, pv, zs = morans_i(coords, z, threshold=MORAN_THRESHOLD,
                          n_perm=MORAN_NPERM, rng=42)
    flag = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
    print(f"  {stat_row['player_name']:<22s}  {len(p):>5d}  {I:>7.4f}  {zs:>7.2f}  {pv:>7.3f}  {flag:>6s}")
    moran_rows.append({"player_name": stat_row["player_name"], "n": len(p),
                        "moran_I": I, "z_score": zs, "p_value": pv})

moran_df = pd.DataFrame(moran_rows)
n_sig = (moran_df["p_value"] < 0.05).sum()
print(f"\n{n_sig}/{len(moran_df)} players have significant spatial clustering (p<0.05)")
if n_sig == 0:
    print("→ No evidence of spatial misspecification. Ellipse geometry looks adequate.")
elif n_sig <= 3:
    print("→ Weak evidence; player-specific issues possible. Shape-ratio plots next.")
else:
    print("→ Widespread spatial clustering. Consider superellipse (Phase 1b).")

# %%
# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("DIAGNOSTIC SUMMARY (2024)")
print("="*65)

mean_resid = preds_2024["residual"].mean()
rmse_all   = np.sqrt((preds_2024["residual"]**2).mean())
print(f"\n1. Global residuals:  mean={mean_resid:+.4f}, RMSE={rmse_all:.4f}")
print(f"   (mean near 0 = unbiased; RMSE near Brier score baseline)")

print(f"\n2. Directional decomp: see bar chart and per-player table above")

bucket_line = " | ".join(
    f"{row.d_bucket}: p̂={row.p_mean:.2f} act={row.catch_rate:.2f}"
    for row in bucket_stats.itertuples()
)
print(f"\n3. Calibration:  {bucket_line}")

print(f"\n4. Shape-ratio vs RMSE: Spearman ρ={rho:.3f} (p={pval:.4f})")
print(f"   (large |ρ| → extreme shapes fit worse → ellipse orientation off)")

print(f"\n5. Moran's I: {n_sig}/{len(moran_df)} players significant at p<0.05")
