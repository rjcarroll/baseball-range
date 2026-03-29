"""
Plotly visualizations for CF range analysis.

Frequentist figures:
  1. plot_player_range()         — catch probability heatmap for one player
  2. plot_range_comparison()     — overlaid reliable-range ellipses for multiple players
  3. plot_rankings()             — horizontal bar chart ranked by ellipse area with bootstrap CIs
  4. plot_opportunity_rankings() — horizontal bar chart ranked by opp-weighted range
  5. plot_spectacular_zone()     — field coverage heatmap

Bayesian figures:
  6. plot_posterior_ellipse()     — posterior-mean heatmap + sample ellipses (credible band)
  7. plot_bayes_rankings()        — rankings with credible interval error bars
  8. plot_season_evolution()      — season-long ellipse area evolution with uncertainty band
  9. plot_prior_posterior_update() — two-panel prior vs. posterior density diagnostic
 10. plot_range_trajectories()    — connected dot plot of a/b posterior means across seasons

Diagnostic figures:
 11. plot_spatial_residuals()    — per-play residuals on (Δx, Δy) field grid
 12. plot_calibration()          — actual catch rate vs. predicted P(catch) by quantile bin
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit

from .data import CF_X0, CF_Y0
from .model import (
    PlayerRange, normalized_distance, catch_probability,
    DEFAULT_BETA_0, DEFAULT_BETA_1,
)


# ── Field geometry helpers ────────────────────────────────────────────────────

def _ellipse_points(a: float, b: float, tau_h: float, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Points tracing the d=1 ellipse (50% catch boundary) at hang time tau_h."""
    theta = np.linspace(0, 2 * np.pi, n)
    x = CF_X0 + a * tau_h * np.cos(theta)
    y = CF_Y0 + b * tau_h * np.sin(theta)
    return x, y


def _asymmetric_ellipse_points(
    a: float,
    b: float,
    gamma: float,
    y0_offset: float,
    tau_h: float,
    n: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Points tracing the d=1 boundary for the v2.0 asymmetric ellipse.

    The boundary is two half-ellipses joined at the lateral axis:
    - Retreating half (dy_adj >= 0): semi-axis b * tau_h
    - Charging half   (dy_adj < 0):  semi-axis b * gamma * tau_h

    Center is at (CF_X0, CF_Y0 + y0_offset).
    """
    cx = CF_X0
    cy = CF_Y0 + y0_offset
    # Retreating half: theta in [0, pi] → sin(theta) >= 0 → dy_adj >= 0
    theta_ret = np.linspace(0, np.pi, n // 2)
    x_ret = cx + a * tau_h * np.cos(theta_ret)
    y_ret = cy + b * tau_h * np.sin(theta_ret)
    # Charging half: theta in [pi, 2pi] → sin(theta) <= 0 → dy_adj <= 0
    theta_chg = np.linspace(np.pi, 2 * np.pi, n // 2)
    x_chg = cx + a * tau_h * np.cos(theta_chg)
    y_chg = cy + b * gamma * tau_h * np.sin(theta_chg)
    return (np.concatenate([x_ret, x_chg]),
            np.concatenate([y_ret, y_chg]))


def _catch_prob_grid(
    a: float,
    b: float,
    tau_h: float = 5.0,
    x_range: tuple = (-130, 130),
    y_range: tuple = (185, 455),
    n: int = 120,
    beta_0: float = DEFAULT_BETA_0,
    beta_1: float = DEFAULT_BETA_1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    dx = X - CF_X0
    dy = Y - CF_Y0
    d = normalized_distance(dx, dy, np.full_like(dx, tau_h), a, b)
    P = expit(beta_0 - beta_1 * d)
    return X, Y, P


def _field_traces() -> list[go.BaseTraceType]:
    """
    Minimal baseball field outlines for the outfield region.
    Distances in feet from home plate using our coordinate system.
    Standard field dimensions: foul lines ~330 ft, CF ~400 ft.
    """
    traces = []

    # Foul lines (approximate)
    for sign in [-1, 1]:
        angle = np.radians(45)
        r = np.linspace(0, 340, 100)
        fx = sign * r * np.sin(angle)
        fy = r * np.cos(angle)
        traces.append(go.Scatter(
            x=fx, y=fy, mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # Warning track arc (approximate, ~20 ft inside fence)
    angles = np.linspace(np.radians(-55), np.radians(55), 200)
    r_fence = 375.0   # approximate CF fence distance
    r_track = r_fence - 20
    traces.append(go.Scatter(
        x=r_track * np.sin(angles),
        y=r_track * np.cos(angles),
        mode="lines",
        line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Fence arc
    traces.append(go.Scatter(
        x=r_fence * np.sin(angles),
        y=r_fence * np.cos(angles),
        mode="lines",
        line=dict(color="rgba(100,100,100,0.5)", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    return traces


# ── Figure 1: Catch probability heatmap ──────────────────────────────────────

def plot_player_range(
    pr: PlayerRange,
    tau_h: float = 5.0,
    title: str | None = None,
) -> go.Figure:
    """
    Catch probability heatmap for a single CF, at a fixed hang time.

    The dashed white contour is the 50% catch boundary (ellipse edge, d=1).
    """
    X, Y, P = _catch_prob_grid(pr.a, pr.b, tau_h=tau_h,
                                beta_0=pr.beta_0, beta_1=pr.beta_1)
    ex, ey = _ellipse_points(pr.a, pr.b, tau_h=tau_h)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=X[0], y=Y[:, 0], z=P,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="P(catch)", tickformat=".0%"),
        hovertemplate="lateral: %{x:.0f} ft<br>depth: %{y:.0f} ft<br>P(catch): %{z:.1%}<extra></extra>",
    ))
    for trace in _field_traces():
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="white", width=2, dash="dash"),
        name="50% boundary",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[CF_X0], y=[CF_Y0], mode="markers",
        marker=dict(size=8, color="white", symbol="x"),
        name="Starting position", hoverinfo="skip",
    ))

    name = pr.player_name or str(pr.player_id)
    fig.update_layout(
        title=title or f"{name} — catch probability (hang time = {tau_h}s)",
        xaxis=dict(title="Lateral (ft, + = RF side)", range=[-130, 130]),
        yaxis=dict(title="Depth from home plate (ft)", range=[185, 455],
                   scaleanchor="x", scaleratio=1),
        width=620, height=600,
        plot_bgcolor="rgba(34,85,34,0.15)",
    )
    return fig


# ── Figure 2: Ellipse overlay comparison ─────────────────────────────────────

def plot_range_comparison(
    results: list[PlayerRange],
    tau_h: float = 5.0,
    title: str | None = None,
    max_players: int = 12,
) -> go.Figure:
    """
    Overlaid 50%-catch-boundary ellipses for multiple CFs.

    Hover each ellipse to see the player name and ellipse area.
    """
    fig = go.Figure()
    for trace in _field_traces():
        fig.add_trace(trace)

    colors = px.colors.qualitative.Plotly
    for i, pr in enumerate(results[:max_players]):
        ex, ey = _ellipse_points(pr.a, pr.b, tau_h=tau_h)
        name = pr.player_name or str(pr.player_id)
        area = pr.ellipse_area(tau_h=tau_h)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"a = {pr.a:.1f} ft/s, b = {pr.b:.1f} ft/s<br>"
                f"ellipse area = {area:,.0f} sq ft<br>"
                f"n = {pr.n_opportunities} opportunities"
                "<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[CF_X0], y=[CF_Y0], mode="markers",
        marker=dict(size=9, color="black", symbol="x"),
        name="Starting position", hoverinfo="skip",
    ))
    fig.update_layout(
        title=title or f"CF Reliable Range Comparison — 50% boundary (hang time = {tau_h}s)",
        xaxis=dict(title="Lateral (ft)", range=[-130, 130]),
        yaxis=dict(title="Depth from home plate (ft)", range=[185, 455],
                   scaleanchor="x", scaleratio=1),
        width=680, height=640,
        plot_bgcolor="rgba(34,85,34,0.15)",
        legend=dict(x=1.02, y=1),
    )
    return fig


# ── Figure 3: Rankings bar chart ──────────────────────────────────────────────

def plot_rankings(
    results: list[PlayerRange],
    tau_h: float = 5.0,
    top_n: int = 20,
    title: str | None = None,
) -> go.Figure:
    """
    Horizontal bar chart ranked by ellipse area (sq ft), with bootstrap ±1 SE bars.

    Ellipse area ∝ π a b τ² — a summary of the player's reliable range at
    a given hang time. Larger = more range.
    """
    top = results[:top_n]
    names = [r.player_name or str(r.player_id) for r in top]
    areas = [r.ellipse_area(tau_h) for r in top]

    # Bootstrap SE for area: propagate via δ(area)/δa and δ(area)/δb
    # area = π a b τ², so σ_area ≈ π τ² sqrt((b σ_a)² + (a σ_b)²)
    area_se = [
        np.pi * tau_h**2 * np.sqrt((r.b * r.a_se) ** 2 + (r.a * r.b_se) ** 2)
        for r in top
    ]

    # Sort by area descending (already sorted, but reverse for horizontal bar)
    order = list(range(len(top) - 1, -1, -1))
    names_sorted = [names[i] for i in order]
    areas_sorted = [areas[i] for i in order]
    se_sorted = [area_se[i] for i in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=areas_sorted, y=names_sorted, orientation="h",
        error_x=dict(type="data", array=se_sorted, visible=True),
        marker_color=px.colors.sequential.Blues[4],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Ellipse area: %{x:,.0f} sq ft<br>"
            "±1 SE: %{error_x.array:,.0f} sq ft"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title or f"CF Range Rankings — ellipse area at hang time = {tau_h}s (top {top_n})",
        xaxis=dict(title="Ellipse area (sq ft)", tickformat=","),
        yaxis=dict(title=""),
        height=max(400, top_n * 28),
        width=650,
        margin=dict(l=160),
    )
    return fig


# ── Figure 4: Opportunity-weighted rankings ───────────────────────────────────

def plot_opportunity_rankings(
    stats_df,
    top_n: int = 20,
    title: str | None = None,
) -> go.Figure:
    """
    Rankings by opportunity-weighted reliable range.

    Stat: fraction of all observed CF fly balls inside the player's R_i.
    This is the preferred single-number ranking for free agent evaluation —
    it weights range by where balls are actually hit.
    """
    import pandas as pd

    top = stats_df.head(top_n).iloc[::-1].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["opp_weighted_range"],
        y=top["player_name"],
        orientation="h",
        marker_color=px.colors.sequential.Blues[4],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Opp-weighted range: %{x:.1%}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title or f"CF Range Rankings — opportunity-weighted reliable range (top {top_n})",
        xaxis=dict(title="Fraction of CF fly balls reliably covered", tickformat=".0%"),
        yaxis=dict(title=""),
        height=max(400, top_n * 28),
        width=680,
        margin=dict(l=160),
    )
    return fig


# ── Figure 5: Spectacular zone coverage heatmap ───────────────────────────────

def plot_spectacular_zone(
    results: list[PlayerRange],
    tau_h: float = 5.0,
    catch_threshold: float = 0.80,
    n_se: float = 1.0,
    x_range: tuple = (-130, 130),
    y_range: tuple = (185, 455),
    n: int = 100,
    title: str | None = None,
) -> go.Figure:
    """
    Field heatmap: how many fielders have reliable range at each location?

    Dark (low coverage) spots are the spectacular zone — where only the best
    CFs can operate. Uses conservative ellipse params (MLE − n_se × SE).
    """
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    dx = (X - CF_X0).ravel()
    dy = (Y - CF_Y0).ravel()
    tau = np.full(len(dx), tau_h)

    coverage = np.zeros(len(dx), dtype=float)
    for pr in results:
        a_cons = max(pr.a - n_se * pr.a_se, 0.5)
        b_cons = max(pr.b - n_se * pr.b_se, 0.5)
        d = normalized_distance(dx, dy, tau, a_cons, b_cons)
        p = catch_probability(d, pr.beta_0, pr.beta_1)
        coverage += (p >= catch_threshold).astype(float)

    Z = coverage.reshape(X.shape)
    n_players = len(results)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorscale="Blues",
        zmin=0, zmax=n_players,
        colorbar=dict(title="# fielders with<br>reliable range"),
        hovertemplate=(
            "lateral: %{x:.0f} ft<br>"
            "depth: %{y:.0f} ft<br>"
            "coverage: %{z:.0f} fielders"
            "<extra></extra>"
        ),
    ))
    for trace in _field_traces():
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=[CF_X0], y=[CF_Y0], mode="markers",
        marker=dict(size=8, color="white", symbol="x"),
        name="Starting position", hoverinfo="skip",
    ))
    fig.update_layout(
        title=title or f"CF Coverage — # fielders with reliable range (hang time = {tau_h}s)",
        xaxis=dict(title="Lateral (ft)", range=list(x_range)),
        yaxis=dict(title="Depth from home plate (ft)", range=list(y_range),
                   scaleanchor="x", scaleratio=1),
        width=640, height=620,
        plot_bgcolor="rgba(34,85,34,0.15)",
    )
    return fig


# ── Figure 6: Bayesian spectacular zone heatmap ──────────────────────────────

def plot_bayes_spectacular_zone(
    posteriors,
    tau_h: float = 5.0,
    catch_threshold: float = 0.80,
    x_range: tuple = (-130, 130),
    y_range: tuple = (185, 455),
    n: int = 100,
    title: str | None = None,
) -> go.Figure:
    """
    Field heatmap: how many fielders have reliable range at each location?

    Uses posterior-mean (a, b) for each player — fast grid computation.
    Dark spots are the spectacular zone where only the best CFs operate.
    """
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    dx = (X - CF_X0).ravel()
    dy = (Y - CF_Y0).ravel()
    tau = np.full(len(dx), tau_h)

    coverage = np.zeros(len(dx), dtype=float)
    for bpr in posteriors:
        d = np.sqrt((dx / (bpr.a_mean * tau)) ** 2 + (dy / (bpr.b_mean * tau)) ** 2)
        from scipy.special import expit
        p = expit(bpr.beta_0 - bpr.beta_1 * d)
        coverage += (p >= catch_threshold).astype(float)

    Z = coverage.reshape(X.shape)
    n_players = len(posteriors)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorscale="Blues",
        zmin=0, zmax=n_players,
        colorbar=dict(title="# fielders with<br>reliable range"),
        hovertemplate=(
            "lateral: %{x:.0f} ft<br>"
            "depth: %{y:.0f} ft<br>"
            "coverage: %{z:.0f} fielders"
            "<extra></extra>"
        ),
    ))
    for trace in _field_traces():
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=[CF_X0], y=[CF_Y0], mode="markers",
        marker=dict(size=8, color="white", symbol="x"),
        name="Starting position", hoverinfo="skip",
    ))
    fig.update_layout(
        title=title or f"CF Coverage — # fielders with reliable range (hang time = {tau_h}s)",
        xaxis=dict(title="Lateral (ft)", range=list(x_range)),
        yaxis=dict(title="Depth from home plate (ft)", range=list(y_range),
                   scaleanchor="x", scaleratio=1),
        width=640, height=620,
        plot_bgcolor="rgba(34,85,34,0.15)",
    )
    return fig


# ── Figure 7: Posterior ellipse with credible band ───────────────────────────

def plot_posterior_ellipse(
    bpr,
    tau_h: float = 5.0,
    n_samples_shown: int = 80,
    title: str | None = None,
) -> go.Figure:
    """
    Catch probability heatmap at posterior-mean (a, b) + credible band.

    N semi-transparent sample ellipses (opacity ~0.06) overlay the
    posterior-mean heatmap, visualizing uncertainty about the ellipse boundary.

    Parameters
    ----------
    bpr : BayesPlayerRange
    n_samples_shown : int
        Number of posterior samples to draw as ellipses (default 80).
        More = denser band but slower render.
    """
    a_mean      = bpr.a_mean
    b_mean      = bpr.b_mean
    y0_offset   = bpr.y0_offset_mean                           # 0.0 for v1
    gamma_mean  = bpr.gamma_mean                               # 1.0 for v1
    is_v2       = bpr.y0_offset_samples is not None

    # Posterior-mean heatmap — v2.0 uses adjusted distance formula
    if is_v2:
        x_range = (-130, 130)
        y_range = (185, 455)
        n_grid  = 120
        xs = np.linspace(*x_range, n_grid)
        ys = np.linspace(*y_range, n_grid)
        Xg, Yg = np.meshgrid(xs, ys)
        dx_g  = Xg - CF_X0
        dy_g  = Yg - CF_Y0
        dy_adj_g      = dy_g - y0_offset
        charge_weight = expit(-dy_adj_g / 5.0)
        b_eff_g       = b_mean * np.exp(np.log(gamma_mean) * charge_weight)
        d_g = np.sqrt((dx_g / (a_mean * tau_h)) ** 2
                      + (dy_adj_g / (b_eff_g * tau_h)) ** 2)
        P = expit(bpr.beta_0 - bpr.beta_1 * d_g)
        X, Y = Xg, Yg
    else:
        X, Y, P = _catch_prob_grid(a_mean, b_mean, tau_h=tau_h,
                                    beta_0=bpr.beta_0, beta_1=bpr.beta_1)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=X[0], y=Y[:, 0], z=P,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="P(catch)", tickformat=".0%"),
        hovertemplate="lateral: %{x:.0f} ft<br>depth: %{y:.0f} ft<br>P(catch): %{z:.1%}<extra></extra>",
    ))
    for trace in _field_traces():
        fig.add_trace(trace)

    # Sample ellipses: credible band
    S = len(bpr.a_samples)
    idx = np.random.default_rng(0).choice(S, size=min(n_samples_shown, S), replace=False)
    for i in idx:
        if is_v2:
            gamma_s = float(np.exp(bpr.log_gamma_samples[i]))
            y0_s    = float(bpr.y0_offset_samples[i])
            ex, ey  = _asymmetric_ellipse_points(
                bpr.a_samples[i], bpr.b_samples[i], gamma_s, y0_s, tau_h=tau_h
            )
        else:
            ex, ey = _ellipse_points(bpr.a_samples[i], bpr.b_samples[i], tau_h=tau_h)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="rgba(255,255,255,0.06)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Posterior-mean ellipse on top
    if is_v2:
        ex_mean, ey_mean = _asymmetric_ellipse_points(
            a_mean, b_mean, gamma_mean, y0_offset, tau_h=tau_h
        )
    else:
        ex_mean, ey_mean = _ellipse_points(a_mean, b_mean, tau_h=tau_h)
    fig.add_trace(go.Scatter(
        x=ex_mean, y=ey_mean, mode="lines",
        line=dict(color="white", width=2.5, dash="dash"),
        name="Posterior mean boundary",
        hoverinfo="skip",
    ))
    # Starting position marker — moves with y0_offset in v2.0
    start_y = CF_Y0 + y0_offset
    fig.add_trace(go.Scatter(
        x=[CF_X0], y=[start_y], mode="markers",
        marker=dict(size=8, color="white", symbol="x"),
        name=f"Starting position ({start_y:.0f} ft)" if is_v2 else "Starting position",
        hoverinfo="skip",
    ))

    name = bpr.player_name or str(bpr.player_id)
    fig.update_layout(
        title=title or f"{name} — posterior catch probability (hang time = {tau_h}s, season {bpr.season})",
        xaxis=dict(title="Lateral (ft, + = RF side)", range=[-130, 130]),
        yaxis=dict(title="Depth from home plate (ft)", range=[185, 455],
                   scaleanchor="x", scaleratio=1),
        width=640, height=620,
        plot_bgcolor="rgba(34,85,34,0.15)",
    )
    return fig


# ── Figure 7: Bayesian rankings with credible intervals ──────────────────────

def plot_bayes_rankings(
    stats_df,
    metric: str = "opp_weighted_range",
    top_n: int = 20,
    title: str | None = None,
) -> go.Figure:
    """
    Horizontal bar chart ranked by a Bayesian statistic, with CI error bars.

    Parameters
    ----------
    stats_df : DataFrame
        Output of bayes_compute_all_stats(). Must have {metric} column.
        For metrics with uncertainty (spectacular_play_prob), also uses
        {metric}_sd for error bars if available.
    metric : str
        Column to rank by. One of 'opp_weighted_range',
        'spectacular_play_prob_mean', 'ellipse_area_5s_mean'.
    """
    top = stats_df.head(top_n).iloc[::-1].reset_index(drop=True)

    sd_col = metric.rstrip("_mean") + "_sd" if metric.endswith("_mean") else metric + "_sd"
    has_sd = sd_col in top.columns

    x_vals = top[metric]
    error_array = top[sd_col] if has_sd else None

    is_pct = metric in ("opp_weighted_range", "spectacular_play_prob_mean")
    tickfmt = ".1%" if is_pct else ","

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals,
        y=top["player_name"],
        orientation="h",
        error_x=(dict(type="data", array=error_array, visible=True) if has_sd else None),
        marker_color=px.colors.sequential.Blues[4],
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{metric}: " + ("%{x:.1%}" if is_pct else "%{x:,.0f}") +
            ("<br>±1 SD: %{error_x.array:.1%}" if has_sd and is_pct else "") +
            "<extra></extra>"
        ),
    ))

    label_map = {
        "opp_weighted_range": "Fraction of CF fly balls reliably covered",
        "spectacular_play_prob_mean": "Catch probability in spectacular zone",
        "ellipse_area_5s_mean": "Ellipse area at 5s hang time (sq ft)",
    }
    x_label = label_map.get(metric, metric)

    season = int(top["season"].iloc[0]) if "season" in top.columns else ""
    fig.update_layout(
        title=title or f"CF Bayesian Rankings — {metric} ({season}, top {top_n})",
        xaxis=dict(title=x_label, tickformat=tickfmt),
        yaxis=dict(title=""),
        height=max(400, top_n * 28),
        width=700,
        margin=dict(l=160),
    )
    return fig


# ── Figure 8: Within-season evolution ────────────────────────────────────────

def plot_season_evolution(
    evolution: list[tuple],
    player_ids: list[int],
    metric: str = "ellipse_area_5s_mean",
    tau_h: float = 5.0,
    title: str | None = None,
) -> go.Figure:
    """
    Line chart showing how a metric evolves through the season.

    Parameters
    ----------
    evolution : list of (cutoff_date, list[BayesPlayerRange])
        Output of fit_season_evolution().
    player_ids : list[int]
        Players to plot. If empty, uses top 5 from final snapshot.
    metric : str
        'ellipse_area_5s_mean' or 'opp_weighted_range'. For ellipse area,
        also draws ±1 SD shading.
    """
    # Collect data per player per cutoff
    records: dict[int, dict] = {}  # pid -> {dates, means, sds, name}

    for cutoff, posteriors in evolution:
        by_id = {bpr.player_id: bpr for bpr in posteriors}

        if not player_ids:
            # Use top 5 from last snapshot by ellipse area
            if cutoff == evolution[-1][0]:
                top5 = sorted(posteriors, key=lambda r: r.ellipse_area(tau_h), reverse=True)[:5]
                player_ids = [r.player_id for r in top5]

        for pid in player_ids:
            if pid not in by_id:
                continue
            bpr = by_id[pid]
            if metric == "ellipse_area_5s_mean":
                areas = np.pi * bpr.a_samples * bpr.b_samples * tau_h ** 2
                val, sd = float(areas.mean()), float(areas.std())
            else:
                val = bpr.ellipse_area(tau_h)
                sd = 0.0

            if pid not in records:
                records[pid] = {"dates": [], "means": [], "sds": [], "name": bpr.player_name}
            records[pid]["dates"].append(cutoff)
            records[pid]["means"].append(val)
            records[pid]["sds"].append(sd)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (pid, data) in enumerate(records.items()):
        color = colors[i % len(colors)]
        dates = data["dates"]
        means = np.array(data["means"])
        sds   = np.array(data["sds"])
        name  = data["name"]

        # Shaded ±1 SD band
        if sds.any():
            fig.add_trace(go.Scatter(
                x=list(dates) + list(reversed(dates)),
                y=list(means + sds) + list(reversed(means - sds)),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.12)"),
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=dates, y=means, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{x|%b %d}<br>"
                f"{metric}: " + ("%{y:,.0f}" if "area" in metric else "%{y:.1%}") +
                "<extra></extra>"
            ),
        ))

    is_area = "area" in metric
    fig.update_layout(
        title=title or f"CF Range — within-season evolution ({metric})",
        xaxis=dict(title="Date", tickformat="%b"),
        yaxis=dict(title="Ellipse area (sq ft)" if is_area else "Fraction covered",
                   tickformat="," if is_area else ".0%"),
        legend=dict(x=1.01, y=1),
        width=780, height=480,
    )
    return fig


# ── Figure 9: Prior vs. posterior diagnostic ─────────────────────────────────

def plot_prior_posterior_update(
    bpr,
    prior_mu_a: float,
    prior_sigma_a: float,
    prior_mu_b: float,
    prior_sigma_b: float,
    title: str | None = None,
) -> go.Figure:
    """
    Two-panel prior vs. posterior density for a single player.

    Visualizes how much data moved the prior. Shrinkage is visible when the
    posterior is narrower than the prior; minimal update = data consistent
    with prior.

    Parameters
    ----------
    bpr : BayesPlayerRange
    prior_mu_a, prior_sigma_a, prior_mu_b, prior_sigma_b : float
        Prior parameters (from build_sequential_priors).
    """
    from scipy.stats import norm as scipy_norm
    from scipy.stats import gaussian_kde

    colors = {"prior": "rgba(150,150,150,0.8)", "posterior": "rgba(31,119,180,0.9)"}

    fig = go.Figure()

    panels = [
        ("a", bpr.a_samples, prior_mu_a, prior_sigma_a, "Lateral speed a (ft/s)"),
        ("b", bpr.b_samples, prior_mu_b, prior_sigma_b, "Depth speed b (ft/s)"),
    ]

    for param, samples, mu, sigma, label in panels:
        lo, hi = mu - 4 * sigma, mu + 4 * sigma
        xs = np.linspace(lo, hi, 300)

        prior_pdf = scipy_norm.pdf(xs, mu, sigma)
        kde = gaussian_kde(samples)
        post_pdf = kde(xs)

        row_suffix = "1" if param == "a" else "2"
        xaxis = "x" if param == "a" else "x2"
        yaxis = "y" if param == "a" else "y2"

        fig.add_trace(go.Scatter(
            x=xs, y=prior_pdf, mode="lines",
            line=dict(color=colors["prior"], width=2, dash="dash"),
            name="Prior" if param == "a" else None,
            showlegend=(param == "a"),
            xaxis=xaxis, yaxis=yaxis,
            hovertemplate=f"Prior — {label}<br>" + "%{x:.1f} ft/s<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=post_pdf, mode="lines",
            line=dict(color=colors["posterior"], width=2),
            name="Posterior" if param == "a" else None,
            showlegend=(param == "a"),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.10)",
            xaxis=xaxis, yaxis=yaxis,
            hovertemplate=f"Posterior — {label}<br>" + "%{x:.1f} ft/s<extra></extra>",
        ))

    name = bpr.player_name or str(bpr.player_id)
    fig.update_layout(
        title=title or f"{name} — prior vs. posterior update (season {bpr.season})",
        xaxis=dict(title="a: lateral speed (ft/s)", domain=[0, 0.47]),
        xaxis2=dict(title="b: depth speed (ft/s)", domain=[0.53, 1.0]),
        yaxis=dict(title="Density"),
        yaxis2=dict(title="Density"),
        width=820, height=400,
        legend=dict(x=0.45, y=1.0),
    )
    return fig


def plot_range_trajectories(
    stats_by_season: dict,
    min_seasons: int = 3,
    min_opps: int = 75,
    top_n: int = 30,
    sort_by: str = "2024",
) -> go.Figure:
    """
    Connected dot plot of lateral (a) and depth (b) posterior means across seasons.

    Each row is a player; dots are posterior means with ±1 SD whiskers; lines
    connect seasons left to right. Two subplots: left panel = a (lateral speed),
    right panel = b (depth speed).

    Parameters
    ----------
    stats_by_season : dict
        Mapping season (int) -> stats DataFrame from bayes_compute_all_stats().
    min_seasons : int
        Minimum number of seasons a player must appear in to be included.
    min_opps : int
        Minimum opportunities per season to count that season.
    top_n : int
        Number of players to show (selected by sort_by season opp_weighted_range).
    sort_by : str or int
        Season to sort players by. Use "2024" (or the latest available season).
    """
    import pandas as pd

    seasons = sorted(stats_by_season.keys())

    # Build long-form table: one row per (player, season)
    rows = []
    for season, df in stats_by_season.items():
        for _, r in df.iterrows():
            if r["n_opportunities"] >= min_opps:
                rows.append({
                    "player_id": r["player_id"],
                    "player_name": r["player_name"],
                    "season": season,
                    "a_mean": r["a_mean"],
                    "a_sd": r.get("a_sd", 0.0),
                    "b_mean": r["b_mean"],
                    "b_sd": r.get("b_sd", 0.0),
                    "opp_weighted_range": r["opp_weighted_range"],
                    "n_opportunities": r["n_opportunities"],
                })
    long = pd.DataFrame(rows)

    # Keep players with enough seasons
    counts = long.groupby("player_id")["season"].count()
    valid_ids = counts[counts >= min_seasons].index
    long = long[long["player_id"].isin(valid_ids)]

    # Select top_n by opp_weighted_range in sort_by season
    sort_season = int(sort_by) if str(sort_by).isdigit() else seasons[-1]
    sort_df = long[long["season"] == sort_season].nlargest(top_n, "opp_weighted_range")
    selected_ids = sort_df["player_id"].tolist()
    long = long[long["player_id"].isin(selected_ids)]

    # Order players by their sort_by rank (top at top)
    order = sort_df.set_index("player_id")["opp_weighted_range"]
    long["rank"] = long["player_id"].map(order)
    long = long.sort_values("rank", ascending=True)
    player_order = long.drop_duplicates("player_id")["player_name"].tolist()

    season_colors = px.colors.qualitative.Plotly[: len(seasons)]
    color_map = dict(zip(seasons, season_colors))

    fig = go.Figure()

    shown_seasons = set()
    for param, xanchor, col in [("a", "x", 1), ("b", "x2", 2)]:
        mean_col = f"{param}_mean"
        sd_col = f"{param}_sd"
        label = "a: lateral speed (ft/s)" if param == "a" else "b: depth speed (ft/s)"

        for pid, grp in long.groupby("player_id"):
            grp = grp.sort_values("season")
            name = grp["player_name"].iloc[0]
            y_vals = [player_order.index(name)] * len(grp)

            # Connecting line
            fig.add_trace(go.Scatter(
                x=grp[mean_col], y=y_vals,
                mode="lines",
                line=dict(color="lightgray", width=1),
                showlegend=False,
                hoverinfo="skip",
                xaxis=xanchor, yaxis="y",
            ))

            # Dots + whiskers per season
            for _, row in grp.iterrows():
                season = int(row["season"])
                show = (season not in shown_seasons) and (col == 1)
                fig.add_trace(go.Scatter(
                    x=[row[mean_col]],
                    y=[player_order.index(name)],
                    mode="markers",
                    marker=dict(color=color_map[season], size=8),
                    error_x=dict(
                        type="data",
                        array=[row[sd_col]],
                        color=color_map[season],
                        thickness=1.5,
                        width=4,
                    ),
                    name=str(season),
                    legendgroup=str(season),
                    showlegend=show,
                    hovertemplate=(
                        f"<b>{name}</b><br>{season}<br>"
                        f"{param}={row[mean_col]:.1f} ± {row[sd_col]:.1f} ft/s<br>"
                        f"n={int(row['n_opportunities'])}<extra></extra>"
                    ),
                    xaxis=xanchor, yaxis="y",
                ))
                if show:
                    shown_seasons.add(season)

    fig.update_layout(
        title=f"Range parameter trajectories {seasons[0]}–{seasons[-1]} "
              f"(top {top_n} by {sort_season} opp-weighted range, ≥{min_opps} opps/season)",
        xaxis=dict(title="a: lateral speed (ft/s)", domain=[0, 0.47]),
        xaxis2=dict(title="b: depth speed (ft/s)", domain=[0.53, 1.0]),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(player_order))),
            ticktext=player_order,
            autorange="reversed",
        ),
        height=max(400, len(player_order) * 22 + 80),
        width=900,
        legend=dict(title="Season", x=1.01, y=1.0),
        margin=dict(l=140, r=120),
    )
    return fig


def plot_spatial_residuals(
    bpr,
    preds: "pd.DataFrame",
    tau_h: float = 5.0,
    title: str | None = None,
) -> go.Figure:
    """
    Per-play residual scatter on the (Δx, Δy) field grid for one player.

    Each dot is a fly ball; color encodes residual = caught − E[P(catch)].
    Blue = model overpredicts (thought it was catchable, wasn't caught).
    Red  = model underpredicts (didn't expect a catch, ball was caught).
    The posterior-mean ellipse boundary is overlaid at hang time tau_h.

    Parameters
    ----------
    bpr : BayesPlayerRange
    preds : DataFrame
        Output of bayes_per_play_predictions() — must include delta_x, delta_y,
        hang_time, caught, p_catch_mean, residual.
    tau_h : float
        Hang time used to draw the ellipse boundary (seconds).
    title : str, optional
    """
    import pandas as pd

    ex, ey = _ellipse_points(bpr.a_mean, bpr.b_mean, tau_h)

    fig = go.Figure()
    fig.add_traces(_field_traces())

    # Residual scatter
    fig.add_trace(go.Scatter(
        x=preds["delta_x"],
        y=preds["delta_y"],
        mode="markers",
        marker=dict(
            color=preds["residual"],
            colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#d6604d"]],
            cmin=-1.0,
            cmax=1.0,
            size=5,
            opacity=0.7,
            colorbar=dict(title="residual<br>(caught − p̂)", x=1.02),
        ),
        text=[
            f"Δx={row.delta_x:.0f} ft, Δy={row.delta_y:.0f} ft<br>"
            f"τ={row.hang_time:.1f}s, caught={int(row.caught)}<br>"
            f"p̂={row.p_catch_mean:.2f}, resid={row.residual:+.2f}"
            for row in preds.itertuples()
        ],
        hovertemplate="%{text}<extra></extra>",
        name="plays",
    ))

    # Posterior-mean ellipse boundary
    fig.add_trace(go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name=f"ellipse (τ={tau_h}s)",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=title or f"{bpr.player_name} — spatial residuals (n={len(preds)})",
        xaxis=dict(title="Δx (ft, + = right field)", range=[-110, 110]),
        yaxis=dict(title="Δy (ft, + = away from plate)",
                   range=[190, 460], scaleanchor="x", scaleratio=1),
        width=560, height=560,
        legend=dict(x=1.05, y=1.0),
    )
    return fig


def plot_calibration(
    all_preds: "pd.DataFrame",
    n_bins: int = 10,
    title: str | None = None,
) -> go.Figure:
    """
    Calibration plot: actual catch rate vs. predicted P(catch) across all players.

    Plays are sorted by predicted p_catch_mean and divided into n_bins equal-count
    bins. Each bin's mean predicted probability is plotted against the actual catch
    rate in that bin. A perfectly calibrated model lies on the diagonal.

    Parameters
    ----------
    all_preds : DataFrame
        Concatenated output of bayes_per_play_predictions() for many players.
        Must include p_catch_mean and caught.
    n_bins : int
        Number of quantile bins (default 10).
    title : str, optional
    """
    import pandas as pd

    df = all_preds[["p_catch_mean", "caught"]].dropna().copy()
    df["bin"] = pd.qcut(df["p_catch_mean"], q=n_bins, labels=False, duplicates="drop")
    grouped = df.groupby("bin").agg(
        p_mean=("p_catch_mean", "mean"),
        catch_rate=("caught", "mean"),
        n=("caught", "count"),
    ).reset_index()

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Calibration dots sized by bin count
    fig.add_trace(go.Scatter(
        x=grouped["p_mean"],
        y=grouped["catch_rate"],
        mode="markers+lines",
        marker=dict(
            size=np.sqrt(grouped["n"]) * 1.5,
            color="#1f77b4",
            line=dict(color="white", width=1),
        ),
        text=[f"n={int(r.n)}<br>p̂={r.p_mean:.2f}, actual={r.catch_rate:.2f}"
              for r in grouped.itertuples()],
        hovertemplate="%{text}<extra></extra>",
        name="bins",
    ))

    fig.update_layout(
        title=title or "Calibration: predicted P(catch) vs. actual catch rate",
        xaxis=dict(title="Mean predicted P(catch)", range=[-0.02, 1.02]),
        yaxis=dict(title="Actual catch rate", range=[-0.02, 1.02]),
        width=520, height=520,
    )
    return fig
