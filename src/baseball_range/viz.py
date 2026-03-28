"""
Plotly visualizations for CF range analysis.

Three main figures:
  1. plot_player_range()     — catch probability heatmap for one player
  2. plot_range_comparison() — overlaid reliable-range ellipses for multiple players
  3. plot_rankings()         — horizontal bar chart ranked by ellipse area with bootstrap CIs
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
