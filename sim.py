"""Quasi-steady-state point-mass lap-time simulator.

Track = curvature along a closed loop. Car = grip coefficient, downforce, drag, power.
At each point the car is limited by (a) cornering grip, (b) engine power / traction when
accelerating, (c) braking grip when slowing for the next corner. The speed profile is the
minimum of those three envelopes.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

RHO, G = 1.2, 9.81  # air density kg/m^3, gravity m/s^2


@dataclass(frozen=True)
class Car:
    mu: float = 1.7        # tyre friction coefficient
    cla: float = 4.5       # downforce coefficient x frontal area (m^2)
    cda: float = 1.0       # drag coefficient x frontal area (m^2)
    power_kw: float = 700  # combined engine + ERS
    mass: float = 810      # car + driver + quali fuel (kg)

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Track:
    s: np.ndarray      # distance from start line (m), uniform spacing
    kappa: np.ndarray  # curvature (1/m)
    x: np.ndarray
    y: np.ndarray

    @property
    def ds(self):
        return self.s[1] - self.s[0]

    @property
    def length(self):
        return self.s[-1] + self.ds


def _loop_smooth(a, w):
    """Moving average over w samples (odd) on a closed loop."""
    return np.convolve(np.r_[a[-w:], a, a[:w]], np.ones(w) / w, "same")[w:-w]


def _loop_diff(a, ds):
    return (np.roll(a, -1) - np.roll(a, 1)) / (2 * ds)


def track_from_xy(x, y, ds=5.0, window=30.0):
    """Closed-loop track from raw positions (metres).

    Positions are resampled every `ds` metres and averaged (twice) over `window` metres before
    taking curvature: position noise would otherwise read as hairpins, and a real racing line
    is smoother than the logged path anyway.
    """
    x, y = np.r_[x, x[0]], np.r_[y, y[0]]  # close the loop
    cum = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    n = round(cum[-1] / ds)
    s, ds = np.linspace(0, cum[-1], n, endpoint=False), cum[-1] / n  # spacing must divide the lap exactly
    w = 2 * int(window / ds / 2) + 1
    xs, ys = (_loop_smooth(_loop_smooth(np.interp(s, cum, a), w), w) for a in (x, y))
    dx, dy = _loop_diff(xs, ds), _loop_diff(ys, ds)
    ddx, ddy = _loop_diff(dx, ds), _loop_diff(dy, ds)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
    return Track(s, kappa, xs, ys)


def _grip(car, v):
    return car.mu * (car.mass * G + 0.5 * RHO * car.cla * v**2)


def _long_force(car, v, k):
    """Longitudinal tyre force left over after cornering (friction circle)."""
    fmax = _grip(car, v)
    flat = car.mass * v**2 * k
    return fmax * np.sqrt(max(0.0, 1 - (flat / fmax) ** 2))


def corner_limit(car, kappa):
    """Max speed through each point if only cornering grip mattered."""
    k = np.abs(kappa)
    denom = car.mass * k - car.mu * 0.5 * RHO * car.cla
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(car.mu * car.mass * G / denom)
    return np.where(denom > 0, v, np.inf)


def simulate(car, track):
    """Speed (m/s) at each track point."""
    n, ds = len(track.s), track.ds
    k = np.abs(track.kappa)
    vlim = np.minimum(corner_limit(car, track.kappa), 120.0)  # 120 m/s cap only matters on infinite straights
    start = int(np.argmin(vlim))  # begin at the slowest apex so the closed loop needs no iteration
    order = (start + np.arange(n)) % n
    p = car.power_kw * 1000
    drag = 0.5 * RHO * car.cda

    fwd = np.empty(n)
    fwd[start] = v = vlim[start]
    for i in order[1:]:
        drive = min(p / max(v, 1.0), _long_force(car, v, k[i]))
        fwd[i] = v = min(np.sqrt(max(v**2 + 2 * ds * (drive - drag * v**2) / car.mass, 0.0)), vlim[i])

    bwd = np.empty(n)
    bwd[start] = v = vlim[start]
    for i in order[:0:-1]:
        brake = _long_force(car, v, k[i]) + drag * v**2  # drag helps braking
        bwd[i] = v = min(np.sqrt(v**2 + 2 * ds * brake / car.mass), vlim[i])

    return np.minimum(fwd, bwd)


def lap_time(v, ds):
    return float(np.sum(ds / v))


FIT = ("mu", "cla", "cda", "power_kw")
_SCALE = np.array([1, 1, 1, 100])  # keep power on the same order as the others for the optimiser


def fit(track, v_real, car=Car(), prior=3.0):
    """Car whose simulated speed best matches `v_real` (m/s at each track point), mass held fixed.

    Grip and downforce trade off almost perfectly against each other, so `prior` pulls the
    parameters gently toward `car`; it costs ~1 km/h of fit and keeps them physically plausible.
    """
    x0 = np.array([getattr(car, f) for f in FIT]) / _SCALE

    def residual(p):
        r = simulate(car.with_(**dict(zip(FIT, p * _SCALE))), track) - v_real
        return np.r_[r, prior * np.sqrt(len(r)) * (p / x0 - 1)]

    r = least_squares(residual, x0, bounds=([0.5, 0, 0.3, 2], [4, 15, 4, 12]), diff_step=1e-3)
    return car.with_(**dict(zip(FIT, (r.x * _SCALE).tolist())))
