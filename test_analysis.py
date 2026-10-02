import numpy as np
import pandas as pd

from analysis import minisector_winners, stints, time_delta


def tel(speed, secs):
    d = np.linspace(0, 1000, len(speed))
    return pd.DataFrame({"Distance": d, "Speed": speed, "Time": pd.to_timedelta(secs, unit="s")})


def test_time_delta_constant_gap():
    ref = tel([200] * 11, np.linspace(0, 10, 11))
    cmp = tel([200] * 11, np.linspace(0.5, 10.5, 11))
    assert np.allclose(time_delta(ref, cmp), 0.5)


def test_minisector_winners():
    a = tel([300] * 5 + [100] * 5, range(10))
    b = tel([200] * 10, range(10))
    assert minisector_winners(a, b, n=2).tolist() == [0, 1]


def test_stints():
    laps = pd.DataFrame({"Driver": ["VER"] * 4, "Stint": [1, 1, 2, 2],
                         "Compound": ["SOFT", "SOFT", "HARD", "HARD"], "LapNumber": [1, 2, 3, 4]})
    out = stints(laps)
    assert out[["Compound", "StartLap", "Laps"]].values.tolist() == [["SOFT", 1, 2], ["HARD", 3, 2]]
