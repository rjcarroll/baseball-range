# %% [markdown]
# # 06 — v2.0 Diagnostic Battery
#
# Two questions:
#   1. Did v2.0 fix the v1.0 pathologies?
#   2. What signals remain that point toward v3.0?
#
# ## v1.0 pathologies (reference)
# - 32/35 players significant Moran's I (p<0.05)
# - Charging zone (225–315°) mean residual: −0.062 (systematic overprediction)
# - Boundary zone calibration (p̂=0.2–0.5): predicted 0.36, actual 0.20 (16pp gap)
# - Root cause: fixed y=310 starting position; fielders play at y≈320–330
#
# ## Tests
# | Test | Class | Question |
# |------|-------|---------|
# | T1: Charging zone bias | PRIMARY | Did v2.0 fix the −0.062 residual? |
# | T2: Moran's I | PRIMARY | How much spatial clustering remains? |
# | T3: Calibration | SECONDARY | Is boundary overprediction fixed? |
# | T4: New parameter summaries | SECONDARY | Are y0, gamma sensible? |
# | T5: Identifiability / collinearity | SECONDARY | Are y0 and gamma well-separated? |
# | T6: Year-over-year correlations | PRIMARY v3 | Are parameters stable YoY? |
# | T7: Empirical σ_transition | PRIMARY v3 | Is λ=1.25 well-calibrated? |

# %%
import sys
sys.path.insert(0, "../src")

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats as scipy_stats
from scipy.spatial import distance_matrix

pio.renderers.default = "browser"

from baseball_range.bayes import load_posteriors, bayes_per_play_predictions
from baseball_range.viz import plot_calibration

CACHE_DIR = pathlib.Path("../data")
DOCS_DIR  = pathlib.Path("../docs")

# v1.0 reference values (hardcoded from 05_diagnostics.py results)
V1_CHARGING_RESIDUAL  = -0.062   # sectors 225–315°
V1_RETREATING_RESIDUAL = +0.020  # sectors 0–90°
V1_BOUNDARY_PRED  = 0.36         # p̂=0.2–0.5 bucket, predicted
V1_BOUNDARY_ACTUAL = 0.20        # p̂=0.2–0.5 bucket, actual
V1_YOY_OWR  = (0.727, 0.787)    # min/max across 3 adjacent-season pairs
V1_YOY_B    = (0.761, 0.842)
V1_YOY_A    = (0.565, 0.712)

UPDATE_SEASONS = [2021, 2022, 2023, 2024]
LAM = 1.25  # off-season inflation factor used in pipeline

# %%
# ── Load posteriors, data, and stats ─────────────────────────────────────────

print("Loading posteriors and data...")
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

print(f"Loaded: {UPDATE_SEASONS}")
for s in UPDATE_SEASONS:
    print(f"  {s}: {len(posteriors_by_season[s])} players, "
          f"{len(data_by_season[s]):,} plays")

# %%
# ── Compute per-play predictions (all seasons) ────────────────────────────────

print("\nComputing per-play predictions (all seasons)...")
preds_by_season = {}
for season in UPDATE_SEASONS:
    rows = []
    for bpr in posteriors_by_season[season]:
        p = bayes_per_play_predictions(bpr, data_by_season[season])
        if len(p):
            p["player_name"] = bpr.player_name
            p["player_id"]   = bpr.player_id
            rows.append(p)
    preds_by_season[season] = pd.concat(rows, ignore_index=True)
    print(f"  {season}: {len(preds_by_season[season]):,} plays, "
          f"{preds_by_season[season]['player_id'].nunique()} players")

# Add approach angle to all seasons
for season in UPDATE_SEASONS:
    df = preds_by_season[season]
    theta = np.degrees(np.arctan2(df["delta_y"], df["delta_x"])) % 360
    df["theta_deg"] = theta

# Pooled over all 4 seasons
preds_all = pd.concat(list(preds_by_season.values()), ignore_index=True)

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T1: Charging zone bias resolution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T1: Charging zone bias resolution")
print("="*70)

BIN_EDGES  = np.arange(0, 361, 45)
BIN_LABELS = [f"{int(lo)}–{int(lo+45)}°" for lo in BIN_EDGES[:-1]]

# Compute sector stats for each season and pooled
def sector_stats(preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    preds["sector"] = pd.cut(
        preds["theta_deg"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
    )
    return (
        preds.groupby("sector", observed=True)["residual"]
        .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)), n="count")
        .reset_index()
    )

sector_all = sector_stats(preds_all)

# Charging sectors: 225–315° = labels index 5 (225–270) and 6 (270–315)
charging_mask = sector_all["sector"].astype(str).str.startswith(("225", "270"))
retreating_mask = sector_all["sector"].astype(str).str.startswith(("0–45", "315–360"))
# Note: 0–45° is label index 0; 315–360 is label index 7
retreating_mask = sector_all["sector"].isin(["0–45°", "315–360°"])

charging_mean  = sector_all.loc[charging_mask, "mean"].mean()
retreating_mean = sector_all.loc[retreating_mask, "mean"].mean()

print(f"Charging zone (225–315°) mean residual: {charging_mean:+.4f}  "
      f"(v1.0: {V1_CHARGING_RESIDUAL:+.4f})")
print(f"Retreating zone (0–90°) mean residual:  {retreating_mean:+.4f}  "
      f"(v1.0: {V1_RETREATING_RESIDUAL:+.4f})")

if abs(charging_mean) < 0.02:
    t1_verdict = "FIXED"
elif abs(charging_mean) < 0.04:
    t1_verdict = "PARTIAL"
else:
    t1_verdict = "INVESTIGATE"
print(f"Verdict: {t1_verdict}")

# Figure: bar chart with v1.0 reference annotations
fig_t1 = go.Figure()
colors = ["#d6604d" if v > 0 else "#2166ac" for v in sector_all["mean"]]
fig_t1.add_trace(go.Bar(
    x=sector_all["sector"].astype(str),
    y=sector_all["mean"],
    error_y=dict(type="data", array=sector_all["se"], visible=True),
    marker_color=colors,
    text=[f"n={int(r.n)}" for r in sector_all.itertuples()],
    hovertemplate="%{x}<br>mean resid=%{y:.4f}<br>%{text}<extra></extra>",
    name="v2.0",
))
# v1.0 reference lines for the charging zone columns
for i, lbl in enumerate(BIN_LABELS):
    if lbl.startswith(("225", "270")):
        fig_t1.add_annotation(
            x=lbl, y=V1_CHARGING_RESIDUAL - 0.005,
            text=f"v1.0: {V1_CHARGING_RESIDUAL:+.3f}",
            showarrow=False, font=dict(color="gray", size=10),
            yanchor="top",
        )
fig_t1.add_hline(y=0, line_dash="dot", line_color="gray")
fig_t1.update_layout(
    title=(f"T1: Directional residuals — v2.0 vs v1.0 reference<br>"
           f"Charging ({charging_mean:+.4f}) | Retreating ({retreating_mean:+.4f}) "
           f"| Verdict: {t1_verdict}"),
    xaxis_title="Approach angle θ = atan2(Δy, Δx)",
    yaxis_title="Mean residual (caught − p̂)",
    width=750, height=450,
)
fig_t1.show()
fig_t1.write_html(str(DOCS_DIR / "diag_v2_directional.html"), include_plotlyjs="cdn")
print("Saved diag_v2_directional.html")

# Per-season charging zone for trend
print("\nCharging zone residual by season:")
for season in UPDATE_SEASONS:
    sc = sector_stats(preds_by_season[season])
    cm = sc.loc[sc["sector"].astype(str).str.startswith(("225", "270")), "mean"].mean()
    print(f"  {season}: {cm:+.4f}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T2: Moran's I spatial autocorrelation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T2: Moran's I spatial autocorrelation")
print("="*70)

MORAN_THRESHOLD = 30   # ft
MORAN_NPERM     = 1000
MIN_OPPS_MORAN  = 150

def morans_i(coords, z, threshold=30.0, n_perm=1000, rng=None):
    """Moran's I with permutation test. Returns (I, p_value, z_score)."""
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
    I_perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = rng.permutation(z_c)
        I_perm[k] = (N / W_sum) * (zp @ W @ zp) / (z_c @ z_c)
    p_val   = (I_perm >= I_obs).mean()
    z_score = (I_obs - I_perm.mean()) / I_perm.std()
    return I_obs, p_val, z_score


def quadrant_residuals(preds: pd.DataFrame) -> dict:
    """Mean residual by spatial quadrant (NE, NW, SE, SW)."""
    dx, dy = preds["delta_x"], preds["delta_y"]
    return {
        "NE": preds.loc[(dx >= 0) & (dy >= 0), "residual"].mean(),
        "NW": preds.loc[(dx <  0) & (dy >= 0), "residual"].mean(),
        "SE": preds.loc[(dx >= 0) & (dy <  0), "residual"].mean(),
        "SW": preds.loc[(dx <  0) & (dy <  0), "residual"].mean(),
    }


# Run on 2024 (matches v1.0 comparison); also run on all-seasons pooled per player
season = 2024
posts_map_2024 = {bpr.player_id: bpr for bpr in posteriors_by_season[season]}
stats_2024 = stats_by_season[season]
preds_2024  = preds_by_season[season]

high_power = (
    stats_2024
    .query("n_opportunities >= @MIN_OPPS_MORAN")
    .sort_values("opp_weighted_range", ascending=False)
)

print(f"Running Moran's I on {len(high_power)} players with ≥{MIN_OPPS_MORAN} opps (2024)...")
print(f"{'Player':<24s}  {'n':>5s}  {'I':>7s}  {'z':>7s}  {'p':>7s}  flag  quadrant pattern")
print("─" * 80)

moran_rows = []
for _, row in high_power.iterrows():
    pid = row["player_id"]
    p = preds_2024[preds_2024["player_id"] == pid]
    if len(p) < 50:
        continue
    coords = p[["delta_x", "delta_y"]].to_numpy()
    z = p["residual"].to_numpy()
    I, pv, zs = morans_i(coords, z, threshold=MORAN_THRESHOLD,
                          n_perm=MORAN_NPERM, rng=42)
    flag = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))

    quad = quadrant_residuals(p)
    # Classify pattern
    south_neg = (quad["SE"] < -0.03) or (quad["SW"] < -0.03)
    diagonal_neg_NE = (quad["NE"] < -0.02) and (quad["SW"] < -0.02)
    diagonal_pos_NW = (quad["NW"] > 0.02) and (quad["SE"] > 0.02)
    if diagonal_neg_NE and diagonal_pos_NW:
        pattern = "diagonal"
    elif south_neg and not diagonal_neg_NE:
        pattern = "directional"
    else:
        pattern = "mixed"

    quad_str = " ".join(f"{k}:{v:+.3f}" for k, v in quad.items())
    print(f"  {row['player_name']:<22s}  {len(p):>5d}  {I:>7.4f}  {zs:>7.2f}  {pv:>7.3f}  "
          f"{flag:>4s}  {pattern}  [{quad_str}]")
    moran_rows.append({
        "player_name": row["player_name"], "n": len(p),
        "moran_I": I, "z_score": zs, "p_value": pv,
        "pattern": pattern,
        **{f"quad_{k}": v for k, v in quad.items()},
    })

moran_df = pd.DataFrame(moran_rows)
n_sig = (moran_df["p_value"] < 0.05).sum()
n_total = len(moran_df)
print(f"\n{n_sig}/{n_total} players significant at p<0.05  (v1.0: 32/35)")

if n_sig > 0:
    patterns = moran_df.loc[moran_df["p_value"] < 0.05, "pattern"].value_counts()
    print(f"Pattern breakdown: {dict(patterns)}")
    dominant = patterns.index[0]
    if dominant == "diagonal":
        t2_note = "DIAGONAL — superellipse signal"
    elif dominant == "directional":
        t2_note = "DIRECTIONAL — y0/gamma partially effective but not complete"
    else:
        t2_note = "MIXED — player-specific, not systematic misspecification"
else:
    t2_note = "NONE — spatial misspecification resolved"

if n_sig == 0:
    t2_verdict = f"CLEAN ({n_sig}/{n_total})"
elif n_sig <= 3:
    t2_verdict = f"WEAK ({n_sig}/{n_total})"
else:
    t2_verdict = f"WIDESPREAD ({n_sig}/{n_total})"
print(f"Verdict: {t2_verdict}  |  {t2_note}")

# Figure: bar chart of Moran's I per player, colored by significance
colors_m = ["#d62728" if pv < 0.01 else ("#ff7f0e" if pv < 0.05 else "#aec7e8")
            for pv in moran_df["p_value"]]
fig_t2 = go.Figure(go.Bar(
    x=moran_df["player_name"],
    y=moran_df["moran_I"],
    marker_color=colors_m,
    text=moran_df["pattern"],
    hovertemplate="<b>%{x}</b><br>I=%{y:.4f}<br>%{text}<extra></extra>",
))
fig_t2.add_hline(y=0, line_dash="dot", line_color="gray")
fig_t2.update_layout(
    title=(f"T2: Moran's I — v2.0 (2024, ≥150 opps)<br>"
           f"{n_sig}/{n_total} significant (p<0.05) — was 32/35 in v1.0"),
    xaxis_title="Player",
    yaxis_title="Moran's I",
    xaxis_tickangle=-45,
    width=900, height=480,
)
fig_t2.show()
fig_t2.write_html(str(DOCS_DIR / "diag_v2_morans.html"), include_plotlyjs="cdn")
print("Saved diag_v2_morans.html")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T3: Calibration
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T3: Calibration")
print("="*70)

fig_t3 = plot_calibration(preds_2024, n_bins=10,
                           title="T3: Calibration — v2.0 (2024)")
fig_t3.show()
fig_t3.write_html(str(DOCS_DIR / "diag_v2_calibration.html"), include_plotlyjs="cdn")
print("Saved diag_v2_calibration.html")

preds_2024["d_bucket"] = pd.cut(
    preds_2024["p_catch_mean"],
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=["hard (<0.2)", "mid-low (0.2–0.5)", "mid-high (0.5–0.8)", "easy (>0.8)"],
)
bucket_stats = (
    preds_2024.groupby("d_bucket", observed=True)
    .agg(p_mean=("p_catch_mean", "mean"), catch_rate=("caught", "mean"), n=("caught", "count"))
    .reset_index()
)
print("\nCalibration by bucket (v2.0 vs v1.0 reference):")
print(f"{'Bucket':<22s}  {'p̂ v2.0':>8s}  {'actual':>8s}  {'gap':>7s}  v1.0 ref")
for row in bucket_stats.itertuples():
    gap = row.p_mean - row.catch_rate
    v1_ref = ""
    if "0.2–0.5" in str(row.d_bucket):
        v1_ref = f"(v1.0: p̂={V1_BOUNDARY_PRED:.2f}, act={V1_BOUNDARY_ACTUAL:.2f}, gap={V1_BOUNDARY_PRED-V1_BOUNDARY_ACTUAL:+.2f})"
    print(f"  {str(row.d_bucket):<20s}  {row.p_mean:>8.3f}  {row.catch_rate:>8.3f}  {gap:>+7.3f}  {v1_ref}")

boundary_row = bucket_stats[bucket_stats["d_bucket"].astype(str).str.contains("0.2")]
if len(boundary_row):
    new_gap = (boundary_row["p_mean"] - boundary_row["catch_rate"]).values[0]
    old_gap = V1_BOUNDARY_PRED - V1_BOUNDARY_ACTUAL
    if abs(new_gap) < 0.05:
        t3_verdict = "FIXED"
    elif abs(new_gap) < abs(old_gap) * 0.5:
        t3_verdict = "PARTIAL"
    else:
        t3_verdict = "UNCHANGED"
    print(f"\nBoundary zone gap: v1.0 = {old_gap:+.3f}, v2.0 = {new_gap:+.3f}  → {t3_verdict}")
else:
    t3_verdict = "N/A"

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T4: New parameter summaries
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T4: New parameter summaries (y0_offset, gamma)")
print("="*70)

# Collect all 2024 player summaries
rows_t4 = []
for bpr in posteriors_by_season[2024]:
    if bpr.y0_offset_samples is not None:
        rows_t4.append({
            "player_name":    bpr.player_name,
            "player_id":      bpr.player_id,
            "y0_offset_mean": bpr.y0_offset_mean,
            "y0_offset_sd":   bpr.y0_offset_sd,
            "gamma_mean":     bpr.gamma_mean,
            "gamma_sd":       bpr.gamma_sd,
            "n_opportunities": bpr.n_opportunities,
        })
params_2024 = pd.DataFrame(rows_t4)

print(f"\ny0_offset (starting depth offset from 310 ft):")
print(f"  mean={params_2024['y0_offset_mean'].mean():.2f} ft  "
      f"median={params_2024['y0_offset_mean'].median():.2f} ft  "
      f"sd={params_2024['y0_offset_mean'].std():.2f} ft")
print(f"  (positive = playing deeper than 310; expected ~+10–20 ft)")

print(f"\ngamma (charge/retreat asymmetry):")
print(f"  mean={params_2024['gamma_mean'].mean():.3f}  "
      f"median={params_2024['gamma_mean'].median():.3f}  "
      f"sd={params_2024['gamma_mean'].std():.3f}")
print(f"  (>1 = faster charging than retreating; expected population mean > 1)")

# Top/bottom 5 by y0_offset
print("\nTop 5 y0_offset (deepest pre-play position):")
print(params_2024.nlargest(5, "y0_offset_mean")[
    ["player_name", "y0_offset_mean", "y0_offset_sd", "n_opportunities"]
].to_string(index=False, float_format="%.2f"))

print("\nBottom 5 y0_offset (shallowest pre-play position):")
print(params_2024.nsmallest(5, "y0_offset_mean")[
    ["player_name", "y0_offset_mean", "y0_offset_sd", "n_opportunities"]
].to_string(index=False, float_format="%.2f"))

print("\nTop 5 gamma (best chargers relative to retreat):")
print(params_2024.nlargest(5, "gamma_mean")[
    ["player_name", "gamma_mean", "gamma_sd", "n_opportunities"]
].to_string(index=False, float_format="%.3f"))

# Cross-season stability for y0_offset and gamma
# For players in ≥3 seasons, compute Spearman ρ(y0_{t}, y0_{t+1})
all_params = []
for season in UPDATE_SEASONS:
    for bpr in posteriors_by_season[season]:
        if bpr.y0_offset_samples is not None:
            all_params.append({
                "player_id":      bpr.player_id,
                "player_name":    bpr.player_name,
                "season":         season,
                "y0_offset_mean": bpr.y0_offset_mean,
                "gamma_mean":     bpr.gamma_mean,
                "n_opportunities": bpr.n_opportunities,
            })
params_all = pd.DataFrame(all_params)

print("\nCross-season Spearman ρ for new parameters (players ≥75 opps in both seasons):")
print(f"{'Pair':<12s}  {'y0_offset ρ':>12s}  {'gamma ρ':>10s}  {'n players':>10s}")
for t, t1 in [(2021,2022),(2022,2023),(2023,2024)]:
    a = params_all.query("season == @t and n_opportunities >= 75")
    b = params_all.query("season == @t1 and n_opportunities >= 75")
    merged = a.merge(b, on="player_id", suffixes=("_a","_b"))
    if len(merged) < 5:
        print(f"  {t}→{t1}:  insufficient data (n={len(merged)})")
        continue
    rho_y0,  _  = scipy_stats.spearmanr(merged["y0_offset_mean_a"], merged["y0_offset_mean_b"])
    rho_gam, _  = scipy_stats.spearmanr(merged["gamma_mean_a"],     merged["gamma_mean_b"])
    print(f"  {t}→{t1}:    {rho_y0:>+.3f}         {rho_gam:>+.3f}    n={len(merged)}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T5: Identifiability and collinearity check
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T5: Identifiability and collinearity check")
print("="*70)

# Posterior SD vs n_opportunities for each parameter
params_sd = []
for bpr in posteriors_by_season[2024]:
    if bpr.y0_offset_samples is None:
        continue
    params_sd.append({
        "player_id":        bpr.player_id,
        "n_opportunities":  bpr.n_opportunities,
        "sd_a":             bpr.a_sd,
        "sd_b":             bpr.b_sd,
        "sd_y0":            bpr.y0_offset_sd,
        "sd_log_gamma":     float(np.std(bpr.log_gamma_samples)) if bpr.log_gamma_samples is not None else np.nan,
        "corr_y0_loggamma": float(np.corrcoef(bpr.y0_offset_samples, bpr.log_gamma_samples)[0,1])
                            if bpr.y0_offset_samples is not None and bpr.log_gamma_samples is not None
                            else np.nan,
    })
sd_df = pd.DataFrame(params_sd)

print("\nPosterior SD vs n_opportunities (median SD in n-bins):")
n_bins = pd.cut(sd_df["n_opportunities"], bins=[0,50,100,200,400,1000])
for param in ["sd_a", "sd_b", "sd_y0", "sd_log_gamma"]:
    meds = sd_df.groupby(n_bins, observed=True)[param].median()
    print(f"  {param:<14s}: " + "  ".join(f"{n_bin}:{v:.3f}" for n_bin, v in meds.items()))

# Collinearity: corr(y0_offset_samples, log_gamma_samples) within each player
min_opps_coll = 50
sd_sub = sd_df[sd_df["n_opportunities"] >= min_opps_coll].dropna(subset=["corr_y0_loggamma"])
abs_corr = sd_sub["corr_y0_loggamma"].abs()
median_corr = abs_corr.median()
high_coll = sd_sub[abs_corr > 0.6]

print(f"\nWithin-player posterior corr(y0_offset, log_gamma) for ≥{min_opps_coll} opps:")
print(f"  Median |corr| = {median_corr:.3f}")

if median_corr > 0.5:
    t5_verdict = "CONCERN — collinearity may limit independent identification"
else:
    t5_verdict = "OK"
print(f"  Verdict: {t5_verdict}")

if len(high_coll) > 0:
    print(f"\n  Players with |corr| > 0.6 ({len(high_coll)}):")
    for bpr in posteriors_by_season[2024]:
        if bpr.player_id in high_coll["player_id"].values:
            row = high_coll[high_coll["player_id"] == bpr.player_id].iloc[0]
            print(f"    {bpr.player_name:<24s}  corr={row['corr_y0_loggamma']:+.3f}  "
                  f"n={bpr.n_opportunities}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T6: Year-over-year correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T6: Year-over-year correlations")
print("="*70)

yoy_rows = []
for t, t1 in [(2021,2022),(2022,2023),(2023,2024)]:
    s_a = stats_by_season[t].rename(columns=lambda c: c+"_a" if c not in ("player_id","player_name") else c)
    s_b = stats_by_season[t1].rename(columns=lambda c: c+"_b" if c not in ("player_id","player_name") else c)
    merged = s_a.merge(s_b, on="player_id")
    merged = merged[
        (merged["n_opportunities_a"] >= 75) &
        (merged["n_opportunities_b"] >= 75)
    ]
    if len(merged) < 5:
        print(f"  {t}→{t1}: insufficient data (n={len(merged)})")
        continue
    result = {"pair": f"{t}→{t1}", "n": len(merged)}
    for param in ["a_mean", "b_mean", "y0_offset_mean", "gamma_mean", "opp_weighted_range"]:
        if f"{param}_a" in merged.columns and f"{param}_b" in merged.columns:
            rho, _ = scipy_stats.spearmanr(merged[f"{param}_a"], merged[f"{param}_b"])
            result[param] = rho
    yoy_rows.append(result)

yoy_df = pd.DataFrame(yoy_rows)

print(f"\n{'Pair':<10s}  {'n':>4s}  {'a':>7s}  {'b':>7s}  {'y0':>7s}  {'gamma':>7s}  {'OWR':>7s}")
print("─" * 60)
for row in yoy_df.itertuples():
    a     = f"{row.a_mean:>+.3f}"     if hasattr(row, "a_mean")     and not pd.isna(row.a_mean)     else "  n/a "
    b     = f"{row.b_mean:>+.3f}"     if hasattr(row, "b_mean")     and not pd.isna(row.b_mean)     else "  n/a "
    y0    = f"{row.y0_offset_mean:>+.3f}" if hasattr(row, "y0_offset_mean") and not pd.isna(row.y0_offset_mean) else "  n/a "
    gam   = f"{row.gamma_mean:>+.3f}" if hasattr(row, "gamma_mean") and not pd.isna(row.gamma_mean) else "  n/a "
    owr   = f"{row.opp_weighted_range:>+.3f}" if hasattr(row, "opp_weighted_range") and not pd.isna(row.opp_weighted_range) else "  n/a "
    print(f"  {row.pair:<10s}  {row.n:>4d}  {a}  {b}  {y0}  {gam}  {owr}")

print(f"\nv1.0 reference: a ρ={V1_YOY_A[0]:.3f}–{V1_YOY_A[1]:.3f}  "
      f"b ρ={V1_YOY_B[0]:.3f}–{V1_YOY_B[1]:.3f}  "
      f"OWR ρ={V1_YOY_OWR[0]:.3f}–{V1_YOY_OWR[1]:.3f}")

# Evaluate degradation
owr_vals = yoy_df["opp_weighted_range"].dropna()
if len(owr_vals):
    if owr_vals.min() < V1_YOY_OWR[0] - 0.05:
        t6_verdict = "DEGRADED — new parameters absorbing signal from a/b"
    elif owr_vals.mean() >= (V1_YOY_OWR[0] + V1_YOY_OWR[1]) / 2 - 0.03:
        t6_verdict = "STABLE"
    else:
        t6_verdict = "SLIGHTLY DEGRADED"
    print(f"Verdict: {t6_verdict}")
else:
    t6_verdict = "N/A"

# Figure: grouped bar chart
if len(yoy_df) > 0:
    params_to_plot = [c for c in ["a_mean","b_mean","y0_offset_mean","gamma_mean","opp_weighted_range"]
                      if c in yoy_df.columns]
    fig_t6 = go.Figure()
    for param in params_to_plot:
        fig_t6.add_trace(go.Bar(
            name=param.replace("_mean","").replace("opp_weighted_range","OWR"),
            x=yoy_df["pair"],
            y=yoy_df[param],
            hovertemplate=f"{param}<br>ρ=%{{y:.3f}}<extra></extra>",
        ))
    fig_t6.update_layout(
        title="T6: Year-over-year Spearman ρ by parameter (≥75 opps both seasons)",
        xaxis_title="Season pair",
        yaxis_title="Spearman ρ",
        yaxis_range=[0, 1.05],
        barmode="group",
        width=700, height=440,
    )
    fig_t6.show()
    fig_t6.write_html(str(DOCS_DIR / "diag_v2_yoy.html"), include_plotlyjs="cdn")
    print("Saved diag_v2_yoy.html")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# T7: Empirical σ_transition vs λ=1.25
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("T7: Empirical σ_transition vs λ=1.25")
print("="*70)
print("Method-of-moments: σ_emp² = var(δ) − mean(SE_t² + SE_{t+1}²)")
print(f"Implied by λ={LAM}: σ_impl = sqrt(λ²−1) × mean_posterior_SD ≈ "
      f"{np.sqrt(LAM**2-1):.3f} × mean_post_SD\n")

# Build panel of per-player posterior moments across seasons
panel = []
for season in UPDATE_SEASONS:
    for bpr in posteriors_by_season[season]:
        if bpr.y0_offset_samples is None:
            continue
        log_gamma_sd = float(np.std(bpr.log_gamma_samples)) if bpr.log_gamma_samples is not None else np.nan
        panel.append({
            "player_id": bpr.player_id,
            "season":    season,
            "a_mean":    bpr.a_mean,      "a_sd":  bpr.a_sd,
            "b_mean":    bpr.b_mean,      "b_sd":  bpr.b_sd,
            "y0_mean":   bpr.y0_offset_mean, "y0_sd": bpr.y0_offset_sd,
            "lg_mean":   float(np.mean(bpr.log_gamma_samples)) if bpr.log_gamma_samples is not None else np.nan,
            "lg_sd":     log_gamma_sd,
            "n_opps":    bpr.n_opportunities,
        })
panel_df = pd.DataFrame(panel)

t7_results = {}
sigma_impl_values = {}

for param, mean_col, sd_col in [
    ("a",        "a_mean",  "a_sd"),
    ("b",        "b_mean",  "b_sd"),
    ("y0_offset","y0_mean", "y0_sd"),
    ("log_gamma","lg_mean", "lg_sd"),
]:
    deltas = []
    ses_sq = []
    post_sds = []
    for pid, grp in panel_df.groupby("player_id"):
        grp = grp.sort_values("season")
        if len(grp) < 2:
            continue
        for i in range(len(grp)-1):
            r0, r1 = grp.iloc[i], grp.iloc[i+1]
            if r0["n_opps"] < 50 or r1["n_opps"] < 50:
                continue
            if pd.isna(r0[mean_col]) or pd.isna(r1[mean_col]):
                continue
            deltas.append(r1[mean_col] - r0[mean_col])
            ses_sq.append(r0[sd_col]**2 + r1[sd_col]**2)
            post_sds.append((r0[sd_col] + r1[sd_col]) / 2)

    if len(deltas) < 5:
        print(f"  {param:<12s}: insufficient pairs (n={len(deltas)})")
        continue

    deltas  = np.array(deltas)
    ses_sq  = np.array(ses_sq)
    post_sds = np.array(post_sds)

    sigma_emp_sq = np.var(deltas) - np.mean(ses_sq)
    sigma_emp = np.sqrt(max(sigma_emp_sq, 0))  # floor at 0 (estimation noise can dominate)
    mean_post_sd = post_sds.mean()
    sigma_impl = np.sqrt(LAM**2 - 1) * mean_post_sd

    ratio = sigma_emp / sigma_impl if sigma_impl > 0 else np.nan
    t7_results[param] = ratio
    sigma_impl_values[param] = (sigma_emp, sigma_impl, mean_post_sd, len(deltas))

    print(f"  {param:<12s}: σ_emp={sigma_emp:.4f}  σ_impl={sigma_impl:.4f}  "
          f"ratio={ratio:.2f}  (n_pairs={len(deltas)}, mean_post_SD={mean_post_sd:.4f})")

# Decision gate
if t7_results:
    high_ratios = {k: v for k, v in t7_results.items() if v > 1.3}
    low_ratios  = {k: v for k, v in t7_results.items() if v < 0.8}
    if len(high_ratios) >= 2:
        t7_verdict = f"v3.0 SIGNAL — λ={LAM} too tight for: {list(high_ratios.keys())}"
    elif len(high_ratios) == 1:
        t7_verdict = f"WEAK SIGNAL — one parameter over-inflated: {list(high_ratios.keys())}"
    elif len(low_ratios) >= 2:
        t7_verdict = f"λ TOO LOOSE for: {list(low_ratios.keys())} (harmless, conservative)"
    else:
        t7_verdict = f"λ OK — empirical transitions consistent with λ={LAM}"
    print(f"\nVerdict: {t7_verdict}")
    print(f"Ratios: {', '.join(f'{k}={v:.2f}' for k, v in t7_results.items())}")
else:
    t7_verdict = "N/A"

# Figure
if sigma_impl_values:
    param_labels = list(sigma_impl_values.keys())
    emp_vals  = [sigma_impl_values[p][0] for p in param_labels]
    impl_vals = [sigma_impl_values[p][1] for p in param_labels]
    fig_t7 = go.Figure()
    fig_t7.add_trace(go.Bar(name="σ_emp (data)",  x=param_labels, y=emp_vals,
                            marker_color="#2166ac"))
    fig_t7.add_trace(go.Bar(name=f"σ_impl (λ={LAM})", x=param_labels, y=impl_vals,
                            marker_color="#d6604d"))
    fig_t7.update_layout(
        title=(f"T7: Empirical vs implied σ_transition (λ={LAM})<br>{t7_verdict}"),
        xaxis_title="Parameter",
        yaxis_title="σ_transition",
        barmode="group",
        width=600, height=420,
    )
    fig_t7.show()
    fig_t7.write_html(str(DOCS_DIR / "diag_v2_sigma_transition.html"), include_plotlyjs="cdn")
    print("Saved diag_v2_sigma_transition.html")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY — v2.0")
print("="*70)

print(f"\nT1 Charging zone residual:  {charging_mean:+.4f}  (v1.0: {V1_CHARGING_RESIDUAL:+.4f})"
      f"  → {t1_verdict}")
print(f"T2 Moran's I significant:   {n_sig}/{n_total} players  (v1.0: 32/35)"
      f"  → {t2_verdict}")
print(f"   Pattern: {t2_note}")
print(f"T3 Boundary calibration:    gap={new_gap:+.3f}  (v1.0: {(V1_BOUNDARY_PRED-V1_BOUNDARY_ACTUAL):+.3f})"
      f"  → {t3_verdict}"
      if 'new_gap' in dir() else f"T3 Boundary calibration:    (see plot)  → {t3_verdict}")
print(f"T5 Collinearity:            median |corr(y0, log_gamma)| = {median_corr:.3f}"
      f"  → {t5_verdict}")
print(f"T6 OWR YoY correlation:     {owr_vals.min():.3f}–{owr_vals.max():.3f}"
      f"  (v1.0: {V1_YOY_OWR[0]:.3f}–{V1_YOY_OWR[1]:.3f})  → {t6_verdict}"
      if 'owr_vals' in dir() and len(owr_vals) else f"T6 OWR YoY:  → {t6_verdict}")
print(f"T7 σ_transition ratio:      {', '.join(f'{k}={v:.2f}' for k,v in t7_results.items())}"
      f"  → {t7_verdict}"
      if t7_results else f"T7 σ_transition:  → {t7_verdict}")

print("\n" + "-"*70)
print("Figures saved to docs/:")
print("  diag_v2_directional.html   T1 — charging zone fix")
print("  diag_v2_morans.html        T2 — spatial autocorrelation")
print("  diag_v2_calibration.html   T3 — calibration improvement")
print("  diag_v2_yoy.html           T6 — year-over-year stability")
print("  diag_v2_sigma_transition.html  T7 — v3.0 signal")
