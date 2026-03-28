# data/

Not tracked in git. Populated by `pull_seasons()` in `data.py`.

| File | Contents |
|------|----------|
| `cf_YYYY.parquet` | CF opportunity data for season YYYY (one file per season) |
| `player_ranges.parquet` | Fitted (a, b) parameters and bootstrap SEs per player |

Pull data:
```python
from baseball_range.data import pull_seasons
df = pull_seasons([2021, 2022, 2023, 2024], cache_dir="data")
```
