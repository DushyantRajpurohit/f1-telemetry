import numpy as np

from sim import Car, corner_limit, fit, lap_time, simulate, track_from_xy


def loop(a, b, n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return track_from_xy(a * np.cos(th), b * np.sin(th))


def test_circle_runs_at_corner_limit():
    tr = loop(80, 80)
    assert np.allclose(1 / tr.kappa, 80, rtol=0.02)  # smoothing shrinks it slightly, but evenly
    v = simulate(Car(), tr)
    v_exact = corner_limit(Car(), tr.kappa)
    assert np.allclose(v, v_exact, rtol=0.01)
    assert abs(lap_time(v, tr.ds) - tr.length / v_exact.mean()) < 0.1


def test_fit_recovers_speed_trace():
    tr = loop(1500, 250)
    target = Car(mu=2.0, cla=3.5, cda=1.3, power_kw=550)
    got = fit(tr, simulate(target, tr), prior=0)
    assert np.sqrt(np.mean((simulate(got, tr) - simulate(target, tr)) ** 2)) < 0.3  # m/s
