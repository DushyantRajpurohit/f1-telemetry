"""Pure analysis functions over FastF1 telemetry/laps DataFrames (no I/O, easy to test)."""
import numpy as np
import pandas as pd


def time_delta(ref: pd.DataFrame, cmp: pd.DataFrame) -> np.ndarray:
    """Gap (s) of `cmp` vs `ref` at each of ref's distance points. Positive = cmp is slower."""
    ref_t = ref["Time"].dt.total_seconds().to_numpy()
    cmp_t = cmp["Time"].dt.total_seconds().to_numpy()
    cmp_at_ref = np.interp(ref["Distance"].to_numpy(), cmp["Distance"].to_numpy(), cmp_t)
    return cmp_at_ref - ref_t


def minisector_winners(a: pd.DataFrame, b: pd.DataFrame, n: int = 25) -> np.ndarray:
    """Split the lap into n equal-distance minisectors; return 0/1 per sector for whoever had the higher mean speed."""
    length = min(a["Distance"].max(), b["Distance"].max())
    edges = np.linspace(0, length, n + 1)

    def mean_speed(t):
        idx = np.clip(np.digitize(t["Distance"], edges) - 1, 0, n - 1)
        return pd.Series(t["Speed"].to_numpy()).groupby(idx).mean().reindex(range(n)).to_numpy()

    return (mean_speed(b) > mean_speed(a)).astype(int)


def sector_of(distance: pd.Series, length: float, n: int = 25) -> np.ndarray:
    """Minisector index for each distance point (used to colour the track map)."""
    return np.clip((distance / length * n).astype(int), 0, n - 1).to_numpy()


def stints(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver stint: compound, first lap, number of laps."""
    return (
        laps.dropna(subset=["Stint"])
        .groupby(["Driver", "Stint", "Compound"], as_index=False)
        .agg(StartLap=("LapNumber", "min"), Laps=("LapNumber", "count"))
        .sort_values(["Driver", "Stint"], ignore_index=True)
    )
