"""
CF range model: per-player ellipse parameters estimated via MLE.

Model
-----
For a fly ball with hang time τ and displacement (Δx, Δy) from the canonical
CF starting position:

    d_i = sqrt( (Δx / (a_i τ))² + (Δy / (b_i τ))² )
    P(catch) = σ(β₀ − β₁ d_i)

(a_i, b_i): lateral and depth speed parameters in ft/sec.
(β₀, β₁):  shared logistic shape parameters; estimated from pooled data,
            then fixed when fitting per-player (a_i, b_i).

A ball at normalized distance d_i = 1 is caught with probability σ(β₀ − β₁).
The ellipse boundary (d_i = 1) is the 50%-catchability contour when β₀ = β₁.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm


# ── Defaults ──────────────────────────────────────────────────────────────────
# β₀ = β₁ places the 50% catch boundary at d_i = 1 (the ellipse edge).
# Large values make the transition sharp.
DEFAULT_BETA_0 = 4.0
DEFAULT_BETA_1 = 4.0

# Starting values for (a, b) optimization: reasonable MLB CF range speeds
A0, B0 = 15.0, 18.0  # ft/sec


# ── Core math ─────────────────────────────────────────────────────────────────

def normalized_distance(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Normalized distance from fielder ellipse boundary."""
    return np.sqrt(
        (delta_x / (a * hang_time)) ** 2 + (delta_y / (b * hang_time)) ** 2
    )


def catch_probability(d: np.ndarray, beta_0: float, beta_1: float) -> np.ndarray:
    return expit(beta_0 - beta_1 * d)


# ── MLE ───────────────────────────────────────────────────────────────────────

def _neg_log_likelihood(
    params: np.ndarray,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    caught: np.ndarray,
    beta_0: float,
    beta_1: float,
) -> float:
    a, b = params
    if a < 0.5 or b < 0.5:
        return 1e10
    d = normalized_distance(delta_x, delta_y, hang_time, a, b)
    p = catch_probability(d, beta_0, beta_1)
    eps = 1e-10
    return -float(
        np.sum(caught * np.log(p + eps) + (1 - caught) * np.log(1 - p + eps))
    )


def fit_player(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    caught: np.ndarray,
    beta_0: float = DEFAULT_BETA_0,
    beta_1: float = DEFAULT_BETA_1,
) -> tuple[float, float]:
    """MLE for (a, b) for a single player. Returns (a_hat, b_hat)."""
    result = minimize(
        _neg_log_likelihood,
        x0=[A0, B0],
        args=(delta_x, delta_y, hang_time, caught, beta_0, beta_1),
        method="Nelder-Mead",
        options={"xatol": 0.05, "fatol": 0.005, "maxiter": 5000},
    )
    return float(result.x[0]), float(result.x[1])


def bootstrap_player(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    caught: np.ndarray,
    n_boot: int = 300,
    beta_0: float = DEFAULT_BETA_0,
    beta_1: float = DEFAULT_BETA_1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap uncertainty for (a, b).

    Returns
    -------
    means : (2,) array  —  bootstrap mean of (a, b)
    sds   : (2,) array  —  bootstrap SD of (a, b)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(caught)
    estimates = np.zeros((n_boot, 2))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        estimates[i] = fit_player(
            delta_x[idx], delta_y[idx], hang_time[idx], caught[idx], beta_0, beta_1
        )
    return estimates.mean(axis=0), estimates.std(axis=0)


# ── Player results dataclass ──────────────────────────────────────────────────

@dataclass
class PlayerRange:
    player_id: int
    player_name: str
    a: float          # lateral speed (ft/sec), MLE
    b: float          # depth speed (ft/sec), MLE
    a_se: float       # bootstrap SE
    b_se: float       # bootstrap SE
    n_opportunities: int
    beta_0: float = DEFAULT_BETA_0
    beta_1: float = DEFAULT_BETA_1
    boot_samples: np.ndarray = field(default=None, repr=False)

    def ellipse_area(self, tau_h: float = 5.0) -> float:
        """Area of the d=1 ellipse at hang time tau_h (sq ft)."""
        return np.pi * (self.a * tau_h) * (self.b * tau_h)


# ── Fit all players ───────────────────────────────────────────────────────────

def _fit_player_job(args: tuple) -> tuple:
    """
    Worker function for parallel player fitting.

    Accepts a flat tuple so it is picklable by ProcessPoolExecutor.
    Returns a tuple that fit_all reassembles into a PlayerRange + log line.
    """
    pid, name, dx, dy, tau, y, n_boot, beta_0, beta_1, seed = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    a_hat, b_hat = fit_player(dx, dy, tau, y, beta_0, beta_1)
    means, sds = bootstrap_player(dx, dy, tau, y, n_boot=n_boot,
                                  beta_0=beta_0, beta_1=beta_1, rng=rng)
    return pid, name, len(dx), a_hat, b_hat, sds[0], sds[1], beta_0, beta_1, time.time() - t0


def fit_all(
    df: pd.DataFrame,
    min_opportunities: int = 75,
    n_boot: int = 300,
    beta_0: float = DEFAULT_BETA_0,
    beta_1: float = DEFAULT_BETA_1,
    seed: int = 42,
    n_jobs: int = 1,
) -> list[PlayerRange]:
    """
    Fit model for all players with at least min_opportunities observations.

    Expects columns: player_id, player_name, delta_x, delta_y, hang_time, caught.
    Logs one line per player: name, n, fitted (a, b) with SEs, and per-player time.

    Parameters
    ----------
    n_jobs : int
        Number of parallel worker processes (default 1 = sequential).
        Set to 2 or 3 to parallelize across players; each player's MLE +
        bootstrap is independent.
    """
    rng = np.random.default_rng(seed)

    # Pre-filter to qualifying players so the progress bar and ETA are accurate.
    all_groups = list(df.groupby("player_id"))
    qualifying = [(pid, g) for pid, g in all_groups if len(g) >= min_opportunities]
    print(
        f"{len(qualifying)} qualifying players (≥{min_opportunities} opportunities) "
        f"of {len(all_groups)} total"
    )

    # Build job args: pass numpy arrays (not DataFrames) so pickling is cheap.
    seeds = rng.integers(0, 2**31, len(qualifying))
    job_args = [
        (
            pid,
            group["player_name"].iloc[0] if "player_name" in group.columns else str(pid),
            group["delta_x"].values,
            group["delta_y"].values,
            group["hang_time"].values,
            group["caught"].values,
            n_boot, beta_0, beta_1, int(seeds[i]),
        )
        for i, (pid, group) in enumerate(qualifying)
    ]

    results = []
    t_start = time.time()

    if n_jobs == 1:
        for args in tqdm(job_args, desc="Fitting"):
            out = _fit_player_job(args)
            pid, name, n, a_hat, b_hat, a_se, b_se, b0, b1, elapsed = out
            tqdm.write(
                f"  {name:<26s}  n={n:4d}  "
                f"a={a_hat:5.1f}±{a_se:.1f}  b={b_hat:5.1f}±{b_se:.1f}  "
                f"{elapsed:.1f}s"
            )
            results.append(PlayerRange(
                player_id=int(pid), player_name=name, a=a_hat, b=b_hat,
                a_se=a_se, b_se=b_se, n_opportunities=n, beta_0=b0, beta_1=b1,
            ))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for args in job_args:
                futures[executor.submit(_fit_player_job, args)] = args[1]  # name
            with tqdm(total=len(job_args), desc=f"Fitting ({n_jobs} workers)") as pbar:
                for future in as_completed(futures):
                    pid, name, n, a_hat, b_hat, a_se, b_se, b0, b1, elapsed = future.result()
                    print(
                        f"  {name:<26s}  n={n:4d}  "
                        f"a={a_hat:5.1f}±{a_se:.1f}  b={b_hat:5.1f}±{b_se:.1f}  "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                    results.append(PlayerRange(
                        player_id=int(pid), player_name=name, a=a_hat, b=b_hat,
                        a_se=a_se, b_se=b_se, n_opportunities=n, beta_0=b0, beta_1=b1,
                    ))
                    pbar.update(1)

    print(f"\nFitted {len(results)} players in {time.time() - t_start:.0f}s")
    return sorted(results, key=lambda r: r.ellipse_area(), reverse=True)


# ── MLE identification filter ─────────────────────────────────────────────────
#
# The MLE optimizer can drift (a, b) to physically implausible values when the
# likelihood surface is flat in one direction — typically for players whose
# observed opportunities cluster tightly along the depth axis (few lateral
# opportunities), so a is essentially unobserved. The optimizer finds a ridge
# where any large value of a fits equally well, and Nelder-Mead wanders far
# from the truth.
#
# This is a fundamental MLE limitation: without prior information, an
# unobserved parameter has no anchor. The Bayesian hierarchical model handles
# this naturally — the population prior on (a, b) prevents drift and shrinks
# thin-data players toward the population mean.
#
# Filter criterion: a or b > MAX_PLAUSIBLE_SPEED (ft/s). No outfielder can
# cover ground at 50 ft/s (~34 mph) — faster than a world-class sprinter.
# Values above this are optimization artifacts, not estimates.

MAX_PLAUSIBLE_SPEED = 50.0  # ft/s — physical upper bound for (a, b)


def filter_identified(results: list[PlayerRange]) -> tuple[list[PlayerRange], list[PlayerRange]]:
    """
    Separate identified from unidentified MLE estimates.

    Returns (identified, unidentified). Unidentified players have at least one
    parameter above MAX_PLAUSIBLE_SPEED — the MLE optimizer drifted to an
    implausible value because the likelihood was flat in that direction.

    These players should be excluded from MLE-based rankings. They are not
    excluded from the Bayesian pipeline, which regularizes them via the
    population prior.
    """
    identified = [r for r in results if r.a < MAX_PLAUSIBLE_SPEED and r.b < MAX_PLAUSIBLE_SPEED]
    unidentified = [r for r in results if r not in identified]
    return identified, unidentified


def results_to_df(results: list[PlayerRange]) -> pd.DataFrame:
    """Convert list of PlayerRange to a tidy DataFrame."""
    return pd.DataFrame([
        {
            "player_id": r.player_id,
            "player_name": r.player_name,
            "a": r.a,
            "b": r.b,
            "a_se": r.a_se,
            "b_se": r.b_se,
            "ellipse_area_5s": r.ellipse_area(tau_h=5.0),
            "n_opportunities": r.n_opportunities,
        }
        for r in results
    ])


# ── Derived statistics ────────────────────────────────────────────────────────

def reliable_range_indicator(
    pr: PlayerRange,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    catch_threshold: float = 0.80,
    n_se: float = 1.0,
) -> np.ndarray:
    """
    Boolean mask: is each play inside player i's reliable range?

    Uses conservative parameters (MLE − n_se × SE) so that R_i covers
    locations the player makes with high probability *and* high confidence.
    """
    a_cons = max(pr.a - n_se * pr.a_se, 0.5)
    b_cons = max(pr.b - n_se * pr.b_se, 0.5)
    d = normalized_distance(delta_x, delta_y, hang_time, a_cons, b_cons)
    p = catch_probability(d, pr.beta_0, pr.beta_1)
    return p >= catch_threshold


def opportunity_weighted_range(
    pr: PlayerRange,
    df: pd.DataFrame,
    catch_threshold: float = 0.80,
    n_se: float = 1.0,
) -> float:
    """
    Fraction of observed CF fly balls inside player i's reliable range.

    Approximates ∫_{R_i} f(x_l, y_l) dx dy against the empirical distribution.
    A higher value means the player reliably converts more of the balls
    actually hit to CF.
    """
    indicator = reliable_range_indicator(
        pr,
        df["delta_x"].values,
        df["delta_y"].values,
        df["hang_time"].values,
        catch_threshold=catch_threshold,
        n_se=n_se,
    )
    return float(indicator.mean())


def compute_spectacular_zone(
    results: list[PlayerRange],
    df: pd.DataFrame,
    max_coverage: int = 1,
    catch_threshold: float = 0.80,
    n_se: float = 1.0,
) -> np.ndarray:
    """
    Boolean mask: which plays land in the spectacular zone S?

    S = plays where at most max_coverage fielders have them in R_j.
    At max_coverage=0 no fielder reliably covers the location;
    max_coverage=1 allows one fielder.
    """
    coverage = np.zeros(len(df), dtype=int)
    for pr in results:
        coverage += reliable_range_indicator(
            pr,
            df["delta_x"].values,
            df["delta_y"].values,
            df["hang_time"].values,
            catch_threshold=catch_threshold,
            n_se=n_se,
        ).astype(int)
    return coverage <= max_coverage


def spectacular_play_prob(
    pr: PlayerRange,
    df: pd.DataFrame,
    spectacular_mask: np.ndarray,
) -> float:
    """
    Opportunity-weighted catch probability in the spectacular zone S.

    ∫_S P(catch | α_i) f(x,y) dx dy  /  ∫_S f(x,y) dx dy

    Uses MLE parameters (not conservative) — we want the player's actual
    expected performance on the hard plays.
    """
    df_s = df[spectacular_mask]
    if len(df_s) == 0:
        return 0.0
    d = normalized_distance(
        df_s["delta_x"].values,
        df_s["delta_y"].values,
        df_s["hang_time"].values,
        pr.a,
        pr.b,
    )
    return float(catch_probability(d, pr.beta_0, pr.beta_1).mean())


def compute_all_stats(
    results: list[PlayerRange],
    df: pd.DataFrame,
    catch_threshold: float = 0.80,
    n_se: float = 1.0,
    max_spectacular_coverage: int = 1,
) -> pd.DataFrame:
    """
    Compute both decision-relevant statistics for every player in results.

    Parameters
    ----------
    results : list[PlayerRange]
        Fitted player results from fit_all().
    df : pd.DataFrame
        Full CF opportunity dataset (all players). Used as the empirical
        ball distribution for opportunity weighting.
    catch_threshold : float
        P(catch) threshold defining reliable range (default 0.80).
    n_se : float
        Conservative ellipse: a_cons = a − n_se × a_se (default 1.0).
    max_spectacular_coverage : int
        A play is in S if covered by at most this many fielders.

    Returns
    -------
    DataFrame with columns: player_id, player_name, a, b, a_se, b_se,
        ellipse_area_5s, opp_weighted_range, spectacular_play_prob,
        n_opportunities. Sorted by opp_weighted_range descending.
    """
    spec_mask = compute_spectacular_zone(
        results, df,
        max_coverage=max_spectacular_coverage,
        catch_threshold=catch_threshold,
        n_se=n_se,
    )
    rows = []
    for pr in tqdm(results, desc="Computing stats"):
        rows.append({
            "player_id": pr.player_id,
            "player_name": pr.player_name,
            "a": pr.a,
            "b": pr.b,
            "a_se": pr.a_se,
            "b_se": pr.b_se,
            "ellipse_area_5s": pr.ellipse_area(5.0),
            "opp_weighted_range": opportunity_weighted_range(
                pr, df, catch_threshold=catch_threshold, n_se=n_se
            ),
            "spectacular_play_prob": spectacular_play_prob(pr, df, spec_mask),
            "n_opportunities": pr.n_opportunities,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("opp_weighted_range", ascending=False)
        .reset_index(drop=True)
    )
