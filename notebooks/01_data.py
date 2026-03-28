# %% [markdown]
# # 01 — Data Pull and Inspection
#
# Pull Statcast fly-ball data for CF opportunities, inspect the raw dataset,
# and cache it for downstream notebooks.
#
# **Runtime:** ~10–20 min per season (Baseball Savant rate limits).
# Cached parquets in `data/` are used on subsequent runs.

# %%
import sys
sys.path.insert(0, "../src")

import pandas as pd
import numpy as np
import plotly.express as px
from baseball_range.data import pull_seasons, load_cf_opportunities, add_player_names, CF_X0, CF_Y0

# %%
# Pull 2021–2024 (caches to data/cf_YYYY.parquet)
df = pull_seasons([2021, 2022, 2023, 2024], cache_dir="../data")
print(f"{len(df):,} CF opportunities across {df['player_id'].nunique()} players")

# %%
# Basic summary
print(df.dtypes)
df.head()

# %%
# Add player names (requires network call to lookup table)
df = add_player_names(df)
df.head()

# %%
# Opportunities per player (top 30)
opp = (df.groupby(["player_id", "player_name"])
         .agg(n=("caught", "count"), catch_rate=("caught", "mean"))
         .sort_values("n", ascending=False)
         .head(30))
print(opp.to_string())

# %%
# Hang time distribution
fig = px.histogram(df, x="hang_time", nbins=50,
                   title="Hang time distribution (CF fly balls)",
                   labels={"hang_time": "Hang time (seconds)"})
fig.show()

# %%
# Landing spot scatter: caught vs. not
sample = df.sample(min(5000, len(df)), random_state=42)
fig = px.scatter(
    sample, x="feet_x", y="feet_y", color="caught",
    color_discrete_map={0: "red", 1: "steelblue"},
    opacity=0.4, size_max=4,
    labels={"feet_x": "Lateral (ft)", "feet_y": "Depth (ft)", "caught": "Caught"},
    title="CF fly-ball landing spots (sample)",
)
fig.add_scatter(x=[CF_X0], y=[CF_Y0], mode="markers",
                marker=dict(size=12, color="black", symbol="x"),
                name="CF starting position")
fig.update_yaxes(range=[185, 455])
fig.update_xaxes(range=[-130, 130])
fig.show()

# %%
# Displacement distributions
fig = px.histogram(df, x="delta_x", nbins=60, title="Δx from canonical position (ft)")
fig.show()
fig = px.histogram(df, x="delta_y", nbins=60, title="Δy from canonical position (ft)")
fig.show()
