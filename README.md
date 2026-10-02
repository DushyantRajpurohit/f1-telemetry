# 🏎️ F1 Telemetry Explorer

Web app that compares any two Formula 1 drivers using real car telemetry from the official F1 live-timing feed (via [FastF1](https://github.com/theOehrly/Fast-F1)), plus a lap-time simulator fitted to real laps.

Pick a season, Grand Prix, session and two drivers to see:

- **Speed trace** of both fastest laps with corner numbers, plus throttle and brake
- **Live time delta**: where on the lap one driver gains or loses time
- **Track dominance map**: the circuit split into 25 minisectors, coloured by the faster driver
- **Lap stats**: top speed, average speed, full-throttle %, gear changes
- **Race pace** lap by lap and **tyre strategy** for the whole grid (race/sprint sessions)

The **Lap simulator** tab fits a point-mass car model to a driver's fastest lap, then lets you change grip, downforce, drag, power and mass to see what each is worth in lap time.

<!-- Add a screenshot: docs/screenshot.png -->

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python server.py               # then open http://localhost:8000
```

The first load of a session downloads ~50–100 MB of timing data; it's cached in `cache/` afterwards.

## Tests

```bash
pytest
```

The analysis and simulator logic are tested offline with synthetic telemetry and tracks.

## How it works

| File | Role |
|------|------|
| `analysis.py` | Time delta by distance interpolation, minisector dominance, stint/tyre aggregation |
| `sim.py` | Point-mass lap simulator: track curvature from X/Y, speed profile, car fitting |
| `server.py` | Starlette JSON API: FastF1 session loading/caching, input validation |
| `static/index.html` | Frontend (vanilla JS + Plotly, no build step) |
| `test_analysis.py`, `test_sim.py` | Offline unit tests |

**Time delta:** both laps' telemetry is indexed by distance; the comparison driver's elapsed time is interpolated onto the reference driver's distance points and subtracted.

**Track dominance:** the lap is cut into equal-distance minisectors; the driver with the higher mean speed in each one wins it, and the track outline (from car X/Y position data) is coloured by the winner.

**Simulator:** positions are resampled every 5 m and smoothed, curvature is taken by finite differences, and the car's speed is the minimum of its cornering limit, a forward acceleration pass (power/traction, friction circle) and a backward braking pass. `scipy.optimize.least_squares` fits grip, downforce, drag and power to the real speed trace, with a mild pull toward typical F1 values because grip and downforce otherwise trade off almost perfectly. Typical fit: 11–19 km/h RMS speed error, lap time within ~1 s.

## Tech

Python · FastF1 · pandas · NumPy · SciPy · Starlette · Plotly · pytest

---
Unofficial project, not associated with Formula 1 companies. F1 data © Formula One World Championship Limited.
