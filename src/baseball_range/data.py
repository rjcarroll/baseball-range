"""
Statcast data pull and preprocessing for CF range analysis.

Coordinate system: feet relative to home plate.
  x: lateral (positive = first-base / RF side)
  y: depth from home plate (positive = outfield)

Home plate sits at approximately pixel (125, 203) in Statcast's 250×250
field image, with roughly 2.5 feet per pixel based on field geometry.
"""

import pathlib
import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_reverse_lookup

# ── Coordinate constants ──────────────────────────────────────────────────────

HP_X_PX = 125.0    # home plate pixel x
HP_Y_PX = 203.0    # home plate pixel y
FEET_PER_PIXEL = 2.5

# Physics constants
MPH_TO_FPS = 1.46667  # mph → ft/s
G_FPS2 = 32.174       # gravitational acceleration (ft/s²)

# Canonical CF starting position (feet from home plate).
# ~310 ft is typical for a standard defensive alignment.
CF_X0 = 0.0
CF_Y0 = 310.0

# CF territory filter bounds (feet)
CF_LAT_MAX = 100.0   # |lateral| < 100 ft
CF_DEPTH_MIN = 200.0
CF_DEPTH_MAX = 450.0


# ── Coordinate helpers ────────────────────────────────────────────────────────

def compute_hang_time(launch_speed_mph: pd.Series, launch_angle_deg: pd.Series) -> pd.Series:
    """
    Derive hang time (seconds) from Statcast launch speed and launch angle.

    Uses the vacuum projectile formula: t = 2 · v₀ · sin(θ) / g
    This is a standard approximation in baseball analytics. Drag shortens
    range but affects hang time less, so this is a reasonable proxy.
    """
    v0 = launch_speed_mph * MPH_TO_FPS
    theta = np.radians(launch_angle_deg)
    return 2.0 * v0 * np.sin(theta) / G_FPS2


def pixels_to_feet(hc_x: pd.Series, hc_y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert Statcast pixel coordinates to feet relative to home plate."""
    feet_x = (hc_x - HP_X_PX) * FEET_PER_PIXEL
    feet_y = (HP_Y_PX - hc_y) * FEET_PER_PIXEL
    return feet_x, feet_y


# ── Data pull ─────────────────────────────────────────────────────────────────

def pull_cf_opportunities(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull Statcast data and filter to CF range opportunities.

    Parameters
    ----------
    start_date, end_date : str
        Date strings in 'YYYY-MM-DD' format.

    Returns
    -------
    DataFrame with columns:
        game_date, player_id, delta_x, delta_y, hang_time, caught,
        feet_x, feet_y
    """
    raw = statcast(start_dt=start_date, end_dt=end_date)

    # Fly balls only (not line drives, popups, ground balls)
    flies = raw[raw["bb_type"] == "fly_ball"].copy()

    # Drop rows missing key fields
    required = ["hc_x", "hc_y", "launch_speed", "launch_angle", "fielder_8", "events"]
    flies = flies.dropna(subset=required)

    # Derive hang time from launch kinematics
    flies["hang_time"] = compute_hang_time(flies["launch_speed"], flies["launch_angle"])

    # Coordinate transform
    flies["feet_x"], flies["feet_y"] = pixels_to_feet(flies["hc_x"], flies["hc_y"])

    # CF territory filter
    cf_mask = (
        (flies["feet_x"].abs() < CF_LAT_MAX)
        & (flies["feet_y"] > CF_DEPTH_MIN)
        & (flies["feet_y"] < CF_DEPTH_MAX)
    )
    cf = flies[cf_mask].copy()

    # Displacement from canonical starting position
    cf["delta_x"] = cf["feet_x"] - CF_X0
    cf["delta_y"] = cf["feet_y"] - CF_Y0

    # Outcome: caught (fly ball out) vs. not caught (hit or error).
    # Statcast codes fly ball outs as "field_out" or "sac_fly".
    cf["caught"] = cf["events"].isin(["field_out", "sac_fly"]).astype(int)

    # CF player ID
    cf["player_id"] = cf["fielder_8"].astype(int)

    keep = ["game_date", "player_id", "delta_x", "delta_y", "hang_time", "caught",
            "feet_x", "feet_y"]
    return cf[keep].reset_index(drop=True)


def pull_seasons(seasons: list[int], cache_dir: str | None = None) -> pd.DataFrame:
    """
    Pull full-season Statcast data, optionally caching to parquet.

    pybaseball is slow for full seasons; caching avoids re-pulling.
    """
    frames = []
    for season in seasons:
        if cache_dir is not None:
            path = pathlib.Path(cache_dir) / f"cf_{season}.parquet"
            if path.exists():
                print(f"Loading {season} from cache...")
                frames.append(pd.read_parquet(path))
                continue

        print(f"Pulling {season} from Baseball Savant (slow)...")
        df = pull_cf_opportunities(f"{season}-04-01", f"{season}-10-01")

        if cache_dir is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"  Cached to {path}")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_cf_opportunities(cache_dir: str = "data") -> pd.DataFrame:
    """Load all cached season parquets from data/."""
    paths = sorted(pathlib.Path(cache_dir).glob("cf_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No cached data found in {cache_dir}/. "
            "Run pull_seasons() first."
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def add_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """Join MLBAM player names onto a DataFrame with a player_id column."""
    ids = df["player_id"].unique().tolist()
    names = playerid_reverse_lookup(ids, key_type="mlbam")[["key_mlbam", "name_last", "name_first"]]
    names["player_name"] = names["name_first"] + " " + names["name_last"]
    names = names.rename(columns={"key_mlbam": "player_id"})
    return df.merge(names[["player_id", "player_name"]], on="player_id", how="left")
