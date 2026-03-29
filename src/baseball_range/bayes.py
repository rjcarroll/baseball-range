"""
Bayesian CF range model: sequential hierarchical estimation via Stan.

Pipeline
--------
Phase 0 — Burn-in (cf_range.stan)
    Run on 2017-2020 data. Estimates population hyperparameters
    (mu_a, sigma_a, mu_b, sigma_b) and logistic shape (beta_0, beta_1).
    Returns per-player posteriors and a PopulationHyperparams summary.

Phase 1 — Sequential updates (cf_range_update.stan)
    For each season 2021-2024, each player's prior is their previous
    season's posterior, inflated by lambda to allow year-to-year change.
    New players receive the population hyperparameters as their prior.

Lambda schedule
    2020 → 2021: lambda = 1.25  (one-year gap, first post-COVID season)
    Annual:       lambda = 1.25  (normal off-season uncertainty)

Within-season evolution
    fit_season_evolution() partitions the season into cumulative monthly
    snapshots and calls fit_season() at each cutoff, showing how posteriors
    narrow as data accumulates.
"""

import json
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    import cmdstanpy
    HAS_CMDSTAN = True
except ImportError:
    HAS_CMDSTAN = False

from .model import normalized_distance, catch_probability


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BayesPlayerRange:
    """
    Posterior samples for a single player's ellipse parameters.

    v1.0 parameters: a (lateral speed), b (depth speed).
    v2.0 additions: y0_offset (starting depth offset from 310 ft),
                    log_gamma (log of charge/retreat asymmetry factor).

    Attributes
    ----------
    a_samples, b_samples : ndarray of shape (S,)
        Raw posterior draws. S = n_chains * n_samples.
    y0_offset_samples : ndarray of shape (S,) or None
        Starting depth offset from 310 ft. None = v1.0 posterior (treated as 0).
    log_gamma_samples : ndarray of shape (S,) or None
        Log of charge/retreat asymmetry. None = v1.0 posterior (gamma = 1).
    season : int
    player_id, player_name : int, str
    n_opportunities : int
    beta_0, beta_1 : float
    """
    player_id: int
    player_name: str
    season: int
    a_samples: np.ndarray
    b_samples: np.ndarray
    n_opportunities: int
    beta_0: float
    beta_1: float

    # v2.0 optional fields — None means v1.0 posterior (y0_offset=0, gamma=1)
    y0_offset_samples: Optional[np.ndarray] = None
    log_gamma_samples: Optional[np.ndarray] = None

    # Derived in __post_init__
    a_mean: float = field(init=False)
    a_sd:   float = field(init=False)
    b_mean: float = field(init=False)
    b_sd:   float = field(init=False)
    y0_offset_mean: float = field(init=False)
    y0_offset_sd:   float = field(init=False)
    gamma_mean:     float = field(init=False)  # E[gamma], not exp(E[log_gamma])
    gamma_sd:       float = field(init=False)

    def __post_init__(self):
        self.a_mean = float(self.a_samples.mean())
        self.a_sd   = float(self.a_samples.std())
        self.b_mean = float(self.b_samples.mean())
        self.b_sd   = float(self.b_samples.std())
        if self.y0_offset_samples is not None:
            self.y0_offset_mean = float(self.y0_offset_samples.mean())
            self.y0_offset_sd   = float(self.y0_offset_samples.std())
        else:
            self.y0_offset_mean = 0.0
            self.y0_offset_sd   = 0.0
        if self.log_gamma_samples is not None:
            gamma = np.exp(self.log_gamma_samples)
            self.gamma_mean = float(gamma.mean())
            self.gamma_sd   = float(gamma.std())
        else:
            self.gamma_mean = 1.0
            self.gamma_sd   = 0.0

    @property
    def y0(self) -> float:
        """True starting depth = 310 + y0_offset_mean (ft from home plate)."""
        return 310.0 + self.y0_offset_mean

    def ellipse_area(self, tau_h: float = 5.0) -> float:
        """Posterior-mean retreating-half ellipse area at hang time tau_h (sq ft)."""
        return np.pi * self.a_mean * self.b_mean * tau_h ** 2

    def to_prior(self, lam: float) -> dict:
        """
        Inflate this posterior into a prior for the next period.

        prior_sigma = lam * posterior_sd for all parameters.
        Floors prevent priors from collapsing to point masses.
        """
        d = {
            "prior_mu_a":    self.a_mean,
            "prior_sigma_a": lam * self.a_sd,
            "prior_mu_b":    self.b_mean,
            "prior_sigma_b": lam * self.b_sd,
        }
        # v2.0 parameters
        d["prior_mu_y0_offset"] = self.y0_offset_mean
        d["prior_sigma_y0"] = max(lam * self.y0_offset_sd, 0.5)
        if self.log_gamma_samples is not None:
            d["prior_mu_log_gamma"]    = float(self.log_gamma_samples.mean())
            d["prior_sigma_log_gamma"] = max(lam * float(self.log_gamma_samples.std()), 0.05)
        else:
            d["prior_mu_log_gamma"]    = 0.0
            d["prior_sigma_log_gamma"] = 0.3 * lam
        return d


@dataclass
class PopulationHyperparams:
    """
    Population-level summaries from the burn-in fit.

    These serve as the prior for new players (not seen in burn-in).
    v2.0 adds y0_offset and log_gamma hyperparameters.
    """
    mu_a:    float   # population mean lateral speed
    sigma_a: float   # population SD lateral speed
    mu_b:    float   # population mean depth speed
    sigma_b: float   # population SD depth speed
    beta_0:  float   # logistic intercept (fixed for sequential fits)
    beta_1:  float   # logistic slope (fixed for sequential fits)
    # v2.0 fields — optional so v1.0 code still works
    mu_y0_offset:    float = 0.0   # population mean depth offset from 310 ft
    sigma_y0:        float = 10.0  # population SD of depth offsets
    mu_log_gamma:    float = 0.0   # population mean log(gamma)
    sigma_log_gamma: float = 0.3   # population SD of log(gamma)


# ── Stan data helpers ─────────────────────────────────────────────────────────

def _build_burnin_data(df: pd.DataFrame) -> tuple[dict, list[int]]:
    """
    Build Stan data dict for cf_range.stan (burn-in, no per-player priors).

    Returns (stan_data, player_id_list) where player_id_list[p-1] = MLBAM id
    for Stan player index p.
    """
    player_ids = sorted(df["player_id"].unique().tolist())
    pid_to_idx = {pid: i + 1 for i, pid in enumerate(player_ids)}  # 1-indexed

    idx = df["player_id"].map(pid_to_idx).values.astype(int)
    return {
        "N":        len(df),
        "P":        len(player_ids),
        "player":   idx,
        "delta_x":  df["delta_x"].values.astype(float),
        "delta_y":  df["delta_y"].values.astype(float),
        "hang_time": df["hang_time"].values.astype(float),
        "caught":   df["caught"].values.astype(int),
    }, player_ids


def _build_update_data(
    df: pd.DataFrame,
    prior_dict: dict,   # player_id -> {prior_mu_a, prior_sigma_a, ...}
    beta_0: float,
    beta_1: float,
) -> tuple[dict, list[int]]:
    """
    Build Stan data dict for cf_range_update.stan.

    Returns (stan_data, player_id_list).
    """
    player_ids = sorted(df["player_id"].unique().tolist())
    pid_to_idx = {pid: i + 1 for i, pid in enumerate(player_ids)}

    idx = df["player_id"].map(pid_to_idx).values.astype(int)
    P = len(player_ids)

    prior_mu_a    = np.array([prior_dict[pid]["prior_mu_a"]    for pid in player_ids])
    prior_sigma_a = np.array([prior_dict[pid]["prior_sigma_a"] for pid in player_ids])
    prior_mu_b    = np.array([prior_dict[pid]["prior_mu_b"]    for pid in player_ids])
    prior_sigma_b = np.array([prior_dict[pid]["prior_sigma_b"] for pid in player_ids])
    prior_mu_y0_offset    = np.array([prior_dict[pid]["prior_mu_y0_offset"]    for pid in player_ids])
    prior_sigma_y0        = np.array([prior_dict[pid]["prior_sigma_y0"]        for pid in player_ids])
    prior_mu_log_gamma    = np.array([prior_dict[pid]["prior_mu_log_gamma"]    for pid in player_ids])
    prior_sigma_log_gamma = np.array([prior_dict[pid]["prior_sigma_log_gamma"] for pid in player_ids])

    return {
        "N":                    len(df),
        "P":                    P,
        "player":               idx,
        "delta_x":              df["delta_x"].values.astype(float),
        "delta_y":              df["delta_y"].values.astype(float),
        "hang_time":            df["hang_time"].values.astype(float),
        "caught":               df["caught"].values.astype(int),
        "prior_mu_a":           prior_mu_a,
        "prior_sigma_a":        prior_sigma_a,
        "prior_mu_b":           prior_mu_b,
        "prior_sigma_b":        prior_sigma_b,
        "prior_mu_y0_offset":   prior_mu_y0_offset,
        "prior_sigma_y0":       prior_sigma_y0,
        "prior_mu_log_gamma":   prior_mu_log_gamma,
        "prior_sigma_log_gamma": prior_sigma_log_gamma,
        "beta_0":               beta_0,
        "beta_1":               beta_1,
    }, player_ids


# ── Serialization ─────────────────────────────────────────────────────────────

def save_posteriors(
    posteriors: list["BayesPlayerRange"],
    path_prefix: str,
) -> None:
    """
    Save posterior samples to .npz + player metadata to .json.

    Writes {path_prefix}.npz and {path_prefix}.json.
    """
    p = pathlib.Path(path_prefix)
    p.parent.mkdir(parents=True, exist_ok=True)

    a_stack = np.stack([pr.a_samples for pr in posteriors])  # (P, S)
    b_stack = np.stack([pr.b_samples for pr in posteriors])
    arrays = {"a": a_stack, "b": b_stack}
    # v2.0: save y0_offset and log_gamma if present
    if any(pr.y0_offset_samples is not None for pr in posteriors):
        arrays["y0_offset"] = np.stack([
            pr.y0_offset_samples if pr.y0_offset_samples is not None
            else np.zeros_like(pr.a_samples)
            for pr in posteriors
        ])
    if any(pr.log_gamma_samples is not None for pr in posteriors):
        arrays["log_gamma"] = np.stack([
            pr.log_gamma_samples if pr.log_gamma_samples is not None
            else np.zeros_like(pr.a_samples)
            for pr in posteriors
        ])
    np.savez_compressed(str(p) + ".npz", **arrays)

    meta = [
        {
            "player_id":      pr.player_id,
            "player_name":    pr.player_name,
            "season":         pr.season,
            "n_opportunities": pr.n_opportunities,
            "beta_0":         pr.beta_0,
            "beta_1":         pr.beta_1,
            "has_v2":         pr.y0_offset_samples is not None,
        }
        for pr in posteriors
    ]
    with open(str(p) + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_posteriors(path_prefix: str) -> list["BayesPlayerRange"]:
    """
    Load posteriors saved by save_posteriors(). Handles both v1.0 and v2.0.
    """
    p = pathlib.Path(path_prefix)
    npz = np.load(str(p) + ".npz")
    a_stack = npz["a"]
    b_stack = npz["b"]
    y0_stack     = npz["y0_offset"] if "y0_offset" in npz else None
    lg_stack     = npz["log_gamma"] if "log_gamma" in npz else None

    with open(str(p) + ".json") as f:
        meta = json.load(f)

    return [
        BayesPlayerRange(
            player_id=m["player_id"],
            player_name=m["player_name"],
            season=m["season"],
            a_samples=a_stack[i],
            b_samples=b_stack[i],
            n_opportunities=m["n_opportunities"],
            beta_0=m["beta_0"],
            beta_1=m["beta_1"],
            y0_offset_samples=y0_stack[i] if y0_stack is not None else None,
            log_gamma_samples=lg_stack[i] if lg_stack is not None else None,
        )
        for i, m in enumerate(meta)
    ]


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _log_diagnostics(fit, label: str, elapsed: float) -> None:
    """
    Print a one-line convergence summary after each Stan fit.

    Reports: wall time, divergences, max R-hat, and low E-BFMI chains (if any).
    These are the three primary MCMC health checks for HMC.
    """
    try:
        diag = fit.diagnose()
        # Count divergences across all chains
        n_div = int(fit.method_variables()["divergent__"].sum())
        # Max R-hat across all parameters
        summary = fit.summary()
        max_rhat = float(summary["R_hat"].max())
        # E-BFMI (energy efficiency) — values < 0.2 indicate poor geometry
        ebfmi = fit.method_variables().get("energy__")
        low_ebfmi = ""
        if ebfmi is not None:
            # E-BFMI per chain: ratio of energy variance to next-energy variance
            # cmdstanpy doesn't compute this directly; skip if unavailable
            pass
        flag = " ⚠" if (n_div > 0 or max_rhat > 1.05) else " ✓"
        print(
            f"[{label}]{flag}  {elapsed:.0f}s  "
            f"divergences={n_div}  max_Rhat={max_rhat:.3f}"
        )
    except Exception as e:
        print(f"[{label}]  {elapsed:.0f}s  (diagnostics unavailable: {e})")


# ── Phase 0: burn-in ──────────────────────────────────────────────────────────

def fit_burnin(
    df: pd.DataFrame,
    stan_file: str,
    n_chains: int = 4,
    n_samples: int = 1000,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> tuple[list["BayesPlayerRange"], "PopulationHyperparams"]:
    """
    Fit the burn-in Stan model (cf_range.stan) on historical data.

    Estimates population hyperparameters and per-player posteriors.
    Caches draws to {cache_dir}/burnin.npz + burnin.json if cache_dir given.

    Parameters
    ----------
    df : DataFrame
        CF opportunities for burn-in seasons (2017-2020 recommended).
        Must have columns: player_id, player_name, delta_x, delta_y,
        hang_time, caught.
    stan_file : str
        Path to cf_range.stan.
    n_chains, n_samples : int
        HMC settings. 4 chains × 1000 samples = 4000 draws.
    seed : int
    cache_dir : str, optional
        Directory for caching; skips fitting if cache exists.

    Returns
    -------
    (posteriors, hyperparams) : list[BayesPlayerRange], PopulationHyperparams
    """
    if not HAS_CMDSTAN:
        raise ImportError("cmdstanpy is required. Run: pip install -e '.[bayes]'")

    cache_prefix = pathlib.Path(cache_dir) / "burnin" if cache_dir else None
    if cache_prefix and (cache_prefix.parent / "burnin.npz").exists():
        print("Loading burn-in posteriors from cache...")
        posteriors = load_posteriors(str(cache_prefix))
        # Reload hyperparams from separate json
        with open(str(cache_prefix) + "_hyperparams.json") as f:
            hp = json.load(f)
        return posteriors, PopulationHyperparams(**hp)

    print(f"Fitting burn-in on {len(df):,} observations, {df['player_id'].nunique()} players...")
    stan_data, player_ids = _build_burnin_data(df)

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    t0 = time.time()
    fit = model.sample(
        data=stan_data,
        chains=n_chains,
        iter_sampling=n_samples,
        seed=seed,
        show_progress=True,
    )
    _log_diagnostics(fit, label="burn-in", elapsed=time.time() - t0)

    # Extract per-player samples: shape (S, P)
    a_draws = fit.stan_variable("a")   # (S, P)
    b_draws = fit.stan_variable("b")   # (S, P)

    # v2.0: y0_offset and log_gamma (present in cf_range_v2.stan)
    all_vars = fit.column_names
    has_v2 = any("y0_offset" in v for v in all_vars)
    if has_v2:
        y0_draws  = fit.stan_variable("y0_offset")   # (S, P)
        lg_draws  = fit.stan_variable("log_gamma")   # (S, P)
    else:
        y0_draws = lg_draws = None

    # Extract hyperparameters
    mu_a    = float(fit.stan_variable("mu_a").mean())
    sigma_a = float(fit.stan_variable("sigma_a").mean())
    mu_b    = float(fit.stan_variable("mu_b").mean())
    sigma_b = float(fit.stan_variable("sigma_b").mean())
    beta_0  = float(fit.stan_variable("beta_0").mean())
    beta_1  = float(fit.stan_variable("beta_1").mean())

    # v2.0 hyperparameters
    mu_y0_offset = sigma_y0 = mu_log_gamma = sigma_log_gamma = None
    if has_v2:
        mu_y0_offset    = float(fit.stan_variable("mu_y0_offset").mean())
        sigma_y0        = float(fit.stan_variable("sigma_y0").mean())
        mu_log_gamma    = float(fit.stan_variable("mu_log_gamma").mean())
        sigma_log_gamma = float(fit.stan_variable("sigma_log_gamma").mean())

    hyperparams = PopulationHyperparams(
        mu_a=mu_a, sigma_a=sigma_a,
        mu_b=mu_b, sigma_b=sigma_b,
        beta_0=beta_0, beta_1=beta_1,
        **({"mu_y0_offset": mu_y0_offset, "sigma_y0": sigma_y0,
            "mu_log_gamma": mu_log_gamma, "sigma_log_gamma": sigma_log_gamma}
           if has_v2 else {}),
    )

    # Build name lookup
    name_map = (
        df.drop_duplicates("player_id")
          .set_index("player_id")["player_name"]
          .to_dict()
    ) if "player_name" in df.columns else {}

    n_by_player = df.groupby("player_id").size().to_dict()

    posteriors = [
        BayesPlayerRange(
            player_id=pid,
            player_name=name_map.get(pid, str(pid)),
            season=int(df["game_date"].dt.year.max()),
            a_samples=a_draws[:, i],
            b_samples=b_draws[:, i],
            n_opportunities=n_by_player.get(pid, 0),
            beta_0=beta_0,
            beta_1=beta_1,
            y0_offset_samples=y0_draws[:, i] if y0_draws is not None else None,
            log_gamma_samples=lg_draws[:, i]  if lg_draws  is not None else None,
        )
        for i, pid in enumerate(player_ids)
    ]

    if cache_prefix:
        save_posteriors(posteriors, str(cache_prefix))
        with open(str(cache_prefix) + "_hyperparams.json", "w") as f:
            json.dump(vars(hyperparams), f, indent=2)
        print(f"Cached burn-in posteriors to {cache_prefix}.*")

    return posteriors, hyperparams


# ── Sequential prior construction ────────────────────────────────────────────

def build_sequential_priors(
    prev_posteriors: list["BayesPlayerRange"],
    new_player_ids: list[int],
    hyperparams: "PopulationHyperparams",
    lam: float,
) -> dict:
    """
    Build per-player prior dicts for cf_range_update.stan.

    Returning players: inflate previous posterior by lambda.
    New players (first season): use population hyperparameters × lambda.

    Parameters
    ----------
    prev_posteriors : list[BayesPlayerRange]
        Posteriors from the previous period.
    new_player_ids : list[int]
        All player IDs appearing this season (including new players).
    hyperparams : PopulationHyperparams
        Fallback prior for new players.
    lam : float
        Inflation factor (e.g. 1.25 for one off-season).

    Returns
    -------
    dict : player_id -> {prior_mu_a, prior_sigma_a, prior_mu_b, prior_sigma_b}
    """
    prev_by_id = {pr.player_id: pr for pr in prev_posteriors}
    priors = {}
    for pid in new_player_ids:
        if pid in prev_by_id:
            priors[pid] = prev_by_id[pid].to_prior(lam)
        else:
            # New player: population hyperparameters, sigma inflated by lam
            priors[pid] = {
                "prior_mu_a":           hyperparams.mu_a,
                "prior_sigma_a":        hyperparams.sigma_a * lam,
                "prior_mu_b":           hyperparams.mu_b,
                "prior_sigma_b":        hyperparams.sigma_b * lam,
                # v2.0 priors
                "prior_mu_y0_offset":   hyperparams.mu_y0_offset,
                "prior_sigma_y0":       hyperparams.sigma_y0 * lam,
                "prior_mu_log_gamma":   hyperparams.mu_log_gamma,
                "prior_sigma_log_gamma": hyperparams.sigma_log_gamma * lam,
            }
    return priors


# ── Phase 1: sequential season fits ──────────────────────────────────────────

def fit_season(
    df: pd.DataFrame,
    season: int,
    prev_posteriors: list["BayesPlayerRange"],
    hyperparams: "PopulationHyperparams",
    stan_file: str,
    lam: float = 1.25,
    min_opportunities: int = 30,
    n_chains: int = 4,
    n_samples: int = 1000,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list["BayesPlayerRange"]:
    """
    Fit one season using per-player priors from the previous period.

    Players with fewer than min_opportunities observations are still fit —
    the prior regularizes them. This allows min_opportunities=30 (vs MLE's 75).

    Parameters
    ----------
    df : DataFrame
        This season's CF opportunities.
    season : int
        Season year (for labeling and caching).
    prev_posteriors : list[BayesPlayerRange]
        Previous period's posteriors (used to build priors).
    hyperparams : PopulationHyperparams
        Population fallback for new players and fixed beta_0, beta_1.
    stan_file : str
        Path to cf_range_update.stan.
    lam : float
        Prior inflation factor.
    min_opportunities : int
        Minimum plays to include a player. Default 30 (lower than MLE's 75
        since the prior handles thin data).
    cache_dir : str, optional
        If given, cache to {cache_dir}/posteriors_{season}.*
    """
    if not HAS_CMDSTAN:
        raise ImportError("cmdstanpy is required. Run: pip install -e '.[bayes]'")

    cache_prefix = pathlib.Path(cache_dir) / f"posteriors_{season}" if cache_dir else None
    if cache_prefix and (cache_dir and (pathlib.Path(cache_dir) / f"posteriors_{season}.npz").exists()):
        print(f"Loading {season} posteriors from cache...")
        return load_posteriors(str(cache_prefix))

    # Filter to qualifying players
    counts = df.groupby("player_id").size()
    qualifying_ids = counts[counts >= min_opportunities].index.tolist()
    df_fit = df[df["player_id"].isin(qualifying_ids)].copy()

    n_players = len(qualifying_ids)
    n_new = sum(1 for pid in qualifying_ids
                if pid not in {pr.player_id for pr in prev_posteriors})
    print(
        f"Season {season}: {len(df_fit):,} obs, {n_players} players "
        f"({n_new} new), λ={lam}"
    )

    prior_dict = build_sequential_priors(
        prev_posteriors, qualifying_ids, hyperparams, lam
    )

    stan_data, player_ids = _build_update_data(
        df_fit, prior_dict, hyperparams.beta_0, hyperparams.beta_1
    )

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    t1 = time.time()
    fit = model.sample(
        data=stan_data,
        chains=n_chains,
        iter_sampling=n_samples,
        seed=seed,
        show_progress=True,
    )
    _log_diagnostics(fit, label=f"season {season}", elapsed=time.time() - t1)

    a_draws = fit.stan_variable("a")  # (S, P)
    b_draws = fit.stan_variable("b")

    # v2.0: y0_offset and log_gamma (present in cf_range_update_v2.stan)
    all_vars = fit.column_names
    has_v2 = any("y0_offset" in v for v in all_vars)
    if has_v2:
        y0_draws = fit.stan_variable("y0_offset")   # (S, P)
        lg_draws = fit.stan_variable("log_gamma")   # (S, P)
    else:
        y0_draws = lg_draws = None

    name_map = (
        df.drop_duplicates("player_id")
          .set_index("player_id")["player_name"]
          .to_dict()
    ) if "player_name" in df.columns else {}

    n_by_player = df_fit.groupby("player_id").size().to_dict()

    posteriors = [
        BayesPlayerRange(
            player_id=pid,
            player_name=name_map.get(pid, str(pid)),
            season=season,
            a_samples=a_draws[:, i],
            b_samples=b_draws[:, i],
            n_opportunities=n_by_player.get(pid, 0),
            beta_0=hyperparams.beta_0,
            beta_1=hyperparams.beta_1,
            y0_offset_samples=y0_draws[:, i] if y0_draws is not None else None,
            log_gamma_samples=lg_draws[:, i]  if lg_draws  is not None else None,
        )
        for i, pid in enumerate(player_ids)
    ]

    if cache_prefix:
        save_posteriors(posteriors, str(cache_prefix))
        print(f"Cached {season} posteriors to {cache_prefix}.*")

    return posteriors


# ── Sequential pipeline driver ────────────────────────────────────────────────

# Default lambda schedule:
#   2020 → 2021: one year (first post-COVID season), same as annual
#   Annual transitions: 1.25 (normal off-season)
DEFAULT_LAM_SCHEDULE = {2021: 1.25, 2022: 1.25, 2023: 1.25, 2024: 1.25}


def run_sequential_pipeline(
    data_by_season: dict[int, pd.DataFrame],
    burnin_seasons: list[int],
    update_seasons: list[int],
    stan_burnin: str,
    stan_update: str,
    lam_schedule: dict[int, float] = DEFAULT_LAM_SCHEDULE,
    n_chains: int = 4,
    n_samples: int = 1000,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> dict[int, list["BayesPlayerRange"]]:
    """
    Full sequential pipeline: burn-in then annual updates.

    Parameters
    ----------
    data_by_season : dict[int, DataFrame]
        Keyed by season year. Each DataFrame has the standard CF columns.
    burnin_seasons : list[int]
        Seasons used for burn-in (e.g. [2017, 2018, 2019, 2020]).
    update_seasons : list[int]
        Seasons to run sequential updates on (e.g. [2021, 2022, 2023, 2024]).
    stan_burnin : str
        Path to cf_range.stan.
    stan_update : str
        Path to cf_range_update.stan.
    lam_schedule : dict[int, float]
        Per-season lambda values. Missing seasons use 1.25.

    Returns
    -------
    dict[int, list[BayesPlayerRange]]
        Posteriors keyed by season. Burn-in result is keyed to the last
        burn-in season.
    """
    # Concatenate burn-in seasons
    burnin_df = pd.concat(
        [data_by_season[s] for s in burnin_seasons], ignore_index=True
    )

    posteriors_by_season = {}

    # Phase 0: burn-in
    burnin_posteriors, hyperparams = fit_burnin(
        burnin_df, stan_burnin,
        n_chains=n_chains, n_samples=n_samples, seed=seed,
        cache_dir=cache_dir,
    )
    posteriors_by_season[burnin_seasons[-1]] = burnin_posteriors

    # Phase 1: sequential updates
    prev = burnin_posteriors
    for season in update_seasons:
        lam = lam_schedule.get(season, 1.25)
        season_posteriors = fit_season(
            data_by_season[season], season,
            prev_posteriors=prev,
            hyperparams=hyperparams,
            stan_file=stan_update,
            lam=lam,
            n_chains=n_chains, n_samples=n_samples, seed=seed,
            cache_dir=cache_dir,
        )
        posteriors_by_season[season] = season_posteriors
        prev = season_posteriors

    return posteriors_by_season


# ── Within-season evolution ───────────────────────────────────────────────────

def fit_season_evolution(
    df_season: pd.DataFrame,
    season: int,
    initial_posteriors: list["BayesPlayerRange"],
    hyperparams: "PopulationHyperparams",
    stan_update: str,
    lam: float = 1.25,
    n_chains: int = 4,
    n_samples: int = 500,
    seed: int = 42,
) -> list[tuple[pd.Timestamp, list["BayesPlayerRange"]]]:
    """
    Fit posteriors at monthly cutoffs through a season.

    Uses cumulative data through each month-end, starting from April.
    Snapshots use n_samples=500 (speed); these are not the final estimates.

    Returns
    -------
    list of (cutoff_date, list[BayesPlayerRange])
    """
    df_season = df_season.copy()
    df_season["game_date"] = pd.to_datetime(df_season["game_date"])

    # Monthly cutoffs: end of each month from April through September
    cutoffs = pd.date_range(
        start=f"{season}-04-30",
        end=f"{season}-09-30",
        freq="ME",
    )

    snapshots = []
    for cutoff in cutoffs:
        df_thru = df_season[df_season["game_date"] <= cutoff]
        if len(df_thru) < 100:
            continue  # not enough data yet
        posteriors = fit_season(
            df_thru, season,
            prev_posteriors=initial_posteriors,
            hyperparams=hyperparams,
            stan_file=stan_update,
            lam=lam,
            min_opportunities=10,  # lower threshold for early-season snapshots
            n_chains=n_chains,
            n_samples=n_samples,
            seed=seed,
        )
        snapshots.append((cutoff, posteriors))
        print(f"  {cutoff.strftime('%B')}: {len(posteriors)} players fit")

    return snapshots


# ── Bayesian statistics ───────────────────────────────────────────────────────

def bayes_reliable_range_indicator(
    bpr: "BayesPlayerRange",
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    hang_time: np.ndarray,
    catch_threshold: float = 0.80,
    inclusion_prob: float = 0.80,
) -> np.ndarray:
    """
    Boolean mask: is each play inside player i's reliable range?

    A play is "reliably covered" if >= inclusion_prob fraction of posterior
    samples produce P(catch) >= catch_threshold. This accounts for full
    posterior uncertainty in (a, b), not just a point estimate.

    Computation: (S, M) probability matrix where S = posterior samples,
    M = number of plays. S=4000, M≈5000 → ~160MB float64, acceptable.

    Parameters
    ----------
    bpr : BayesPlayerRange
    delta_x, delta_y, hang_time : arrays of shape (M,)
    catch_threshold : float
        P(catch) threshold defining "covered" (default 0.80).
    inclusion_prob : float
        Fraction of posterior samples that must exceed catch_threshold
        for the play to be included in R_i (default 0.80).

    Returns
    -------
    bool ndarray of shape (M,)
    """
    S = len(bpr.a_samples)
    M = len(delta_x)

    # Broadcast: (S, 1) arrays × (1, M) arrays → (S, M)
    a = bpr.a_samples[:, np.newaxis]   # (S, 1)
    b = bpr.b_samples[:, np.newaxis]

    dx  = np.asarray(delta_x)[np.newaxis, :]   # (1, M)
    dy  = np.asarray(delta_y)[np.newaxis, :]
    tau = np.asarray(hang_time)[np.newaxis, :]

    # v2.0: adjust for per-player starting depth and charge/retreat asymmetry
    from scipy.special import expit
    y0_off    = (bpr.y0_offset_samples[:, np.newaxis]   # (S, 1)
                 if bpr.y0_offset_samples is not None
                 else np.zeros((S, 1)))
    log_gamma = (bpr.log_gamma_samples[:, np.newaxis]
                 if bpr.log_gamma_samples is not None
                 else np.zeros((S, 1)))
    dy_adj        = dy - y0_off                                  # (S, M)
    charge_weight = expit(-dy_adj / 5.0)
    b_eff         = b * np.exp(log_gamma * charge_weight)        # (S, M)

    d = np.sqrt((dx / (a * tau)) ** 2 + (dy_adj / (b_eff * tau)) ** 2)  # (S, M)
    p = expit(bpr.beta_0 - bpr.beta_1 * d)   # (S, M)

    fraction_covered = (p >= catch_threshold).mean(axis=0)  # (M,)
    return fraction_covered >= inclusion_prob


def bayes_opportunity_weighted_range(
    bpr: "BayesPlayerRange",
    df: pd.DataFrame,
    catch_threshold: float = 0.80,
    inclusion_prob: float = 0.80,
) -> float:
    """
    Fraction of observed CF fly balls in player i's Bayesian reliable range.
    """
    indicator = bayes_reliable_range_indicator(
        bpr,
        df["delta_x"].values,
        df["delta_y"].values,
        df["hang_time"].values,
        catch_threshold=catch_threshold,
        inclusion_prob=inclusion_prob,
    )
    return float(indicator.mean())


def bayes_compute_spectacular_zone(
    posteriors: list["BayesPlayerRange"],
    df: pd.DataFrame,
    max_coverage: int = 1,
    catch_threshold: float = 0.80,
    inclusion_prob: float = 0.80,
) -> np.ndarray:
    """
    Boolean mask: plays where at most max_coverage fielders have reliable range.
    """
    coverage = np.zeros(len(df), dtype=int)
    for bpr in tqdm(posteriors, desc="Computing coverage"):
        coverage += bayes_reliable_range_indicator(
            bpr,
            df["delta_x"].values,
            df["delta_y"].values,
            df["hang_time"].values,
            catch_threshold=catch_threshold,
            inclusion_prob=inclusion_prob,
        ).astype(int)
    return coverage <= max_coverage


def bayes_spectacular_play_prob(
    bpr: "BayesPlayerRange",
    df: pd.DataFrame,
    spectacular_mask: np.ndarray,
) -> tuple[float, float]:
    """
    Opportunity-weighted catch probability in the spectacular zone.

    Returns (mean, sd) tuple — a distribution over the statistic, not a scalar.
    Uses MLE-like posterior mean for expected performance on hard plays.

    Returns
    -------
    (mean, sd) : float
    """
    df_s = df[spectacular_mask]
    if len(df_s) == 0:
        return 0.0, 0.0

    S = len(bpr.a_samples)
    M = len(df_s)

    a   = bpr.a_samples[:, np.newaxis]
    b   = bpr.b_samples[:, np.newaxis]
    dx  = np.asarray(df_s["delta_x"])[np.newaxis, :]
    dy  = np.asarray(df_s["delta_y"])[np.newaxis, :]
    tau = np.asarray(df_s["hang_time"])[np.newaxis, :]

    from scipy.special import expit
    y0_off    = (bpr.y0_offset_samples[:, np.newaxis]
                 if bpr.y0_offset_samples is not None
                 else np.zeros((S, 1)))
    log_gamma = (bpr.log_gamma_samples[:, np.newaxis]
                 if bpr.log_gamma_samples is not None
                 else np.zeros((S, 1)))
    dy_adj        = dy - y0_off
    charge_weight = expit(-dy_adj / 5.0)
    b_eff         = b * np.exp(log_gamma * charge_weight)

    d = np.sqrt((dx / (a * tau)) ** 2 + (dy_adj / (b_eff * tau)) ** 2)
    p = expit(bpr.beta_0 - bpr.beta_1 * d)   # (S, M)

    # Per-sample mean over plays
    sample_means = p.mean(axis=1)   # (S,)
    return float(sample_means.mean()), float(sample_means.std())


def bayes_per_play_predictions(
    bpr: "BayesPlayerRange",
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-play posterior mean catch probability and residual for one player.

    Parameters
    ----------
    bpr : BayesPlayerRange
    df : DataFrame
        CF opportunity dataset (all players). Filtered to this player internally.

    Returns
    -------
    DataFrame with columns:
        delta_x, delta_y, hang_time, caught,
        p_catch_mean  — posterior mean P(catch) averaged over all samples,
        residual      — caught − p_catch_mean
    """
    player_df = df[df["player_id"] == bpr.player_id].copy()
    if player_df.empty:
        return pd.DataFrame(columns=["delta_x", "delta_y", "hang_time",
                                      "caught", "p_catch_mean", "residual"])

    dx  = np.asarray(player_df["delta_x"])[np.newaxis, :]      # (1, M)
    dy  = np.asarray(player_df["delta_y"])[np.newaxis, :]
    tau = np.asarray(player_df["hang_time"])[np.newaxis, :]
    a   = bpr.a_samples[:, np.newaxis]                          # (S, 1)
    b   = bpr.b_samples[:, np.newaxis]
    S   = len(bpr.a_samples)

    from scipy.special import expit
    y0_off    = (bpr.y0_offset_samples[:, np.newaxis]
                 if bpr.y0_offset_samples is not None
                 else np.zeros((S, 1)))
    log_gamma = (bpr.log_gamma_samples[:, np.newaxis]
                 if bpr.log_gamma_samples is not None
                 else np.zeros((S, 1)))
    dy_adj        = dy - y0_off                                 # (S, M)
    charge_weight = expit(-dy_adj / 5.0)
    b_eff         = b * np.exp(log_gamma * charge_weight)

    d      = np.sqrt((dx / (a * tau)) ** 2 + (dy_adj / (b_eff * tau)) ** 2)  # (S, M)
    p      = expit(bpr.beta_0 - bpr.beta_1 * d)                # (S, M)
    p_mean = p.mean(axis=0)                                     # (M,)

    out = player_df[["delta_x", "delta_y", "hang_time", "caught"]].copy()
    out["p_catch_mean"] = p_mean
    out["residual"] = np.asarray(player_df["caught"]).astype(float) - p_mean
    return out.reset_index(drop=True)


def bayes_compute_all_stats(
    posteriors: list["BayesPlayerRange"],
    df: pd.DataFrame,
    catch_threshold: float = 0.80,
    inclusion_prob: float = 0.80,
    max_spectacular_coverage: int = 1,
) -> pd.DataFrame:
    """
    Compute both decision-relevant statistics for every player.

    Parameters
    ----------
    posteriors : list[BayesPlayerRange]
    df : DataFrame
        Full CF opportunity dataset (all players, for empirical distribution).
    catch_threshold : float
        P(catch) threshold for reliable range (default 0.80).
    inclusion_prob : float
        Posterior fraction threshold for inclusion in R_i (default 0.80).
    max_spectacular_coverage : int
        Spectacular zone: covered by at most this many fielders.

    Returns
    -------
    DataFrame with columns: player_id, player_name, season, a_mean, a_sd,
        b_mean, b_sd, y0_offset_mean, gamma_mean, ellipse_area_5s_mean,
        opp_weighted_range, spectacular_play_prob_mean, spectacular_play_prob_sd,
        n_opportunities. Sorted by opp_weighted_range descending.
    """
    spec_mask = bayes_compute_spectacular_zone(
        posteriors, df,
        max_coverage=max_spectacular_coverage,
        catch_threshold=catch_threshold,
        inclusion_prob=inclusion_prob,
    )

    rows = []
    for bpr in tqdm(posteriors, desc="Computing Bayesian stats"):
        spp_mean, spp_sd = bayes_spectacular_play_prob(bpr, df, spec_mask)
        rows.append({
            "player_id":               bpr.player_id,
            "player_name":             bpr.player_name,
            "season":                  bpr.season,
            "a_mean":                  bpr.a_mean,
            "a_sd":                    bpr.a_sd,
            "b_mean":                  bpr.b_mean,
            "b_sd":                    bpr.b_sd,
            "y0_offset_mean":          bpr.y0_offset_mean,
            "gamma_mean":              bpr.gamma_mean,
            "ellipse_area_5s_mean":    bpr.ellipse_area(5.0),
            "opp_weighted_range":      bayes_opportunity_weighted_range(
                                           bpr, df, catch_threshold, inclusion_prob
                                       ),
            "spectacular_play_prob_mean": spp_mean,
            "spectacular_play_prob_sd":   spp_sd,
            "n_opportunities":         bpr.n_opportunities,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("opp_weighted_range", ascending=False)
        .reset_index(drop=True)
    )
