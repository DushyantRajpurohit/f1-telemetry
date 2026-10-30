import pytest
from fastf1.exceptions import DataNotLoadedError
from starlette.exceptions import HTTPException

import server


def test_missing_timing_data_explains_the_subscription(monkeypatch):
    def not_loaded(*args):
        raise DataNotLoadedError("not loaded")
    monkeypatch.setattr(server, "_load", not_loaded)

    with pytest.raises(HTTPException) as e:
        server.load(2024, "Monaco Grand Prix", "Q")

    assert e.value.status_code == 503
    assert "F1 TV Access" in e.value.detail


def test_other_load_failures_keep_their_own_message(monkeypatch):
    def broken(*args):
        raise ValueError("no lap data yet")
    monkeypatch.setattr(server, "_load", broken)

    with pytest.raises(HTTPException) as e:
        server.load(2024, "Monaco Grand Prix", "Q")

    assert e.value.status_code == 502
    assert "no lap data yet" in e.value.detail
