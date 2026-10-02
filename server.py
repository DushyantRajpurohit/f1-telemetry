"""F1 Telemetry Explorer API + static frontend. Run: python server.py  (then open http://localhost:8000)"""
import math
import threading
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import sim
from analysis import minisector_winners, sector_of, stints, time_delta

HERE = Path(__file__).parent
(HERE / "cache").mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(HERE / "cache"))
fastf1.set_log_level("WARNING")

KINDS = ("Q", "R", "S", "SQ", "FP1", "FP2", "FP3")
FIRST_YEAR = 2018  # FastF1 telemetry starts here
N_SECTORS = 25
# Allowed what-if ranges: (min, max) per Car field
SIM_RANGES = {"mu": (0.5, 4), "cla": (0, 15), "cda": (0.3, 4), "power_kw": (100, 1500), "mass": (600, 1100)}


def nums(a, digits=1):
    """JSON-safe list: rounded floats, NaN -> null."""
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)
            for v in np.asarray(a, dtype=float).tolist()]


# ---- input validation ---------------------------------------------------------------------

def arg(request, name):
    v = request.query_params.get(name)
    if not v:
        raise HTTPException(400, f"Missing '{name}'")
    return v


def year_arg(request):
    try:
        y = int(arg(request, "year"))
    except ValueError:
        raise HTTPException(400, "Bad year")
    if not FIRST_YEAR <= y <= date.today().year:
        raise HTTPException(400, f"Year must be {FIRST_YEAR}-{date.today().year}")
    return y


def session_args(request):
    year, event, kind = year_arg(request), arg(request, "event"), arg(request, "kind")
    if kind not in KINDS:
        raise HTTPException(400, f"Session must be one of {', '.join(KINDS)}")
    when = {e["name"]: e["date"] for e in schedule(year)}.get(event)
    if when is None:
        raise HTTPException(400, "Unknown Grand Prix for that season")
    if date.fromisoformat(when) - timedelta(days=3) > date.today():
        raise HTTPException(400, f"{event} {year} hasn't happened yet")
    return year, event, kind


def driver_arg(request, name, session):
    d = arg(request, name)
    if d not in set(session.results["Abbreviation"].dropna()):
        raise HTTPException(400, f"{d} didn't take part in this session")
    return d


# ---- FastF1 loading (cached in memory; FastF1 also caches downloads on disk) ---------------

@lru_cache(maxsize=8)
def schedule(year):
    s = fastf1.get_event_schedule(year, include_testing=False)
    return [{"name": r.EventName, "date": r.EventDate.date().isoformat(), "format": r.EventFormat}
            for r in s.itertuples()]


_load_lock = threading.Lock()  # ponytail: one download at a time; per-session locks if many users


@lru_cache(maxsize=4)  # a loaded session is ~100 MB of DataFrames
def _load(year, event, kind):
    s = fastf1.get_session(year, event, kind)
    s.load(weather=False, messages=False)
    if s.laps.empty:
        raise ValueError("no lap data yet")
    return s


def load(year, event, kind):
    with _load_lock:
        try:
            return _load(year, event, kind)
        except Exception as e:  # FastF1 raises a zoo of types for "not available"
            raise HTTPException(502, f"Couldn't load {event} {kind} {year}: {e}")


def fastest(session, drv):
    lap = session.laps.pick_drivers(drv).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        raise HTTPException(422, f"{drv} has no timed lap in this session")
    return lap


def color(drv, session):
    try:
        return fastf1.plotting.get_driver_color(drv, session)
    except Exception:
        return None


# ---- endpoints ----------------------------------------------------------------------------

def events(request):
    return JSONResponse(schedule(year_arg(request)))


def drivers(request):
    s = load(*session_args(request))
    r = s.results.dropna(subset=["Abbreviation"])
    return JSONResponse([{"abbr": d.Abbreviation, "name": d.FullName, "team": d.TeamName,
                          "color": color(d.Abbreviation, s)} for d in r.itertuples()])


def lap_stats(tel):
    return {"Top speed (km/h)": float(tel["Speed"].max()),
            "Avg speed (km/h)": round(float(tel["Speed"].mean()), 1),
            "Full throttle %": round(float((tel["Throttle"] >= 98).mean() * 100), 1),
            "Gear changes": int((tel["nGear"].diff().fillna(0) != 0).sum())}


def compare(request):
    year, event, kind = session_args(request)
    s = load(year, event, kind)
    a, b = driver_arg(request, "a", s), driver_arg(request, "b", s)
    if a == b:
        raise HTTPException(400, "Pick two different drivers")
    lap_a, lap_b = fastest(s, a), fastest(s, b)
    tel_a, tel_b = lap_a.get_telemetry(), lap_b.get_telemetry()

    def trace(lap, tel):
        return {"lapTime": lap["LapTime"].total_seconds(), "distance": nums(tel["Distance"]),
                "speed": nums(tel["Speed"]), "throttle": nums(tel["Throttle"]),
                "brake": nums(tel["Brake"].astype(int), 0), "gear": nums(tel["nGear"], 0),
                "stats": lap_stats(tel)}

    try:
        corners = [{"n": f"{c.Number}{c.Letter or ''}", "d": round(float(c.Distance))}
                   for c in s.get_circuit_info().corners.itertuples()]
    except Exception:
        corners = []  # circuit info missing for some older events

    winners = minisector_winners(tel_a, tel_b, N_SECTORS)
    out = {
        "a": trace(lap_a, tel_a), "b": trace(lap_b, tel_b),
        "delta": nums(time_delta(tel_a, tel_b), 3),
        "corners": corners,
        "map": {"x": nums(tel_a["X"] / 10), "y": nums(tel_a["Y"] / 10),  # FastF1 X/Y are in decimetres
                "owner": winners[sector_of(tel_a["Distance"], tel_a["Distance"].max(), N_SECTORS)].tolist(),
                "won": [int((winners == 0).sum()), int((winners == 1).sum())]},
        "race": None,
    }
    if kind in ("R", "S"):
        pace = {}
        for d in (a, b):
            q = s.laps.pick_drivers(d).pick_quicklaps()
            pace[d] = {"lap": q["LapNumber"].astype(int).tolist(), "time": nums(q["LapTime"].dt.total_seconds(), 3)}
        st = stints(s.laps)
        out["race"] = {"pace": pace,
                       "order": [d for d in s.results["Abbreviation"].dropna() if d in set(st["Driver"])],
                       "stints": [{"driver": r.Driver, "compound": r.Compound, "start": int(r.StartLap), "laps": int(r.Laps)}
                                  for r in st.itertuples()]}
    return JSONResponse(out)


@lru_cache(maxsize=16)
def baseline(year, event, kind, drv):
    """Track + real speed for a driver's fastest lap, and the car fitted to it."""
    lap = fastest(_load(year, event, kind), drv)
    tel, lap_time = lap.get_telemetry(), lap["LapTime"].total_seconds()
    track = sim.track_from_xy(tel["X"].to_numpy() / 10, tel["Y"].to_numpy() / 10)
    v_real = np.interp(track.s * tel["Distance"].max() / track.length, tel["Distance"], tel["Speed"]) / 3.6
    return track, v_real, sim.fit(track, v_real), lap_time


def simulate(request):
    year, event, kind = session_args(request)
    s = load(year, event, kind)
    drv = driver_arg(request, "driver", s)
    track, v_real, fitted, real_time = baseline(year, event, kind, drv)

    overrides = {}
    for k, (lo, hi) in SIM_RANGES.items():
        if k in request.query_params:
            try:
                overrides[k] = float(request.query_params[k])
            except ValueError:
                raise HTTPException(400, f"Bad {k}")
            if not lo <= overrides[k] <= hi:
                raise HTTPException(400, f"{k} must be {lo}-{hi}")
    car = fitted.with_(**overrides)
    v_fit, v_car = sim.simulate(fitted, track), sim.simulate(car, track)
    return JSONResponse({
        "s": nums(track.s, 0), "x": nums(track.x), "y": nums(track.y),
        "real": nums(v_real * 3.6), "fitted": nums(v_fit * 3.6), "car": nums(v_car * 3.6),
        "fittedCar": vars(fitted), "usedCar": vars(car), "ranges": SIM_RANGES,
        "realTime": real_time,
        "fittedTime": round(sim.lap_time(v_fit, track.ds), 3),
        "carTime": round(sim.lap_time(v_car, track.ds), 3),
        "rmse": round(float(np.sqrt(np.mean((v_fit - v_real) ** 2)) * 3.6), 1),
    })


async def http_error(request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app = Starlette(
    routes=[Route("/api/events", events), Route("/api/drivers", drivers),
            Route("/api/compare", compare), Route("/api/sim", simulate),
            Mount("/", StaticFiles(directory=HERE / "static", html=True))],
    exception_handlers={HTTPException: http_error},
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
