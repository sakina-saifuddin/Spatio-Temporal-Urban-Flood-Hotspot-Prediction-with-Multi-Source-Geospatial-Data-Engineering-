"""Policy-conscious public-place geocoding.

Default provider: the public OpenStreetMap Nominatim service. Its usage policy
(https://operations.osmfoundation.org/policies/nominatim/) requires an
identifying User-Agent, at most one request per second application-wide,
visible attribution, no autocomplete and no bulk geocoding. This module:

* only runs after an explicit search submission (no autocomplete);
* serialises requests process-wide with a lock and a 1.1 s minimum interval;
* sends a configurable identifying User-Agent;
* never writes queries to disk or logs (results are held in process memory
  only for de-duplication of identical repeated queries);
* returns structured errors for timeouts, provider errors and empty results.

Configuration (environment variables or Streamlit secrets):
``FLOODSENSE_GEOCODER_URL``, ``FLOODSENSE_GEOCODER_USER_AGENT``,
``FLOODSENSE_GEOCODER_ENABLED`` ("false" disables external search).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import requests

DEFAULT_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "FloodSense-research-app/1.0 (educational flood susceptibility explorer)"
PROVIDER_NAME = "OpenStreetMap Nominatim"
ATTRIBUTION = "Search results © OpenStreetMap contributors (ODbL), via Nominatim."
MIN_INTERVAL_SECONDS = 1.1
TIMEOUT_SECONDS = 8
MAX_QUERY_LENGTH = 200
# Bias results toward the study area (west, north, east, south).
STUDY_VIEWBOX = "-95.85,30.25,-94.95,29.40"

_lock = threading.Lock()
_last_request = [0.0]
_memory_cache: OrderedDict[str, "GeocodeResult"] = OrderedDict()
_CACHE_LIMIT = 128


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    kind: str = ""


@dataclass
class GeocodeResult:
    ok: bool
    places: list[Place] = field(default_factory=list)
    error: str | None = None


def _setting(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets are optional.
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - secrets file absent or not in Streamlit
        pass
    return default


def provider_config() -> dict[str, str | bool]:
    enabled = _setting("FLOODSENSE_GEOCODER_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return {
        "enabled": enabled,
        "url": _setting("FLOODSENSE_GEOCODER_URL", DEFAULT_URL),
        "user_agent": _setting("FLOODSENSE_GEOCODER_USER_AGENT", DEFAULT_USER_AGENT),
        "provider": PROVIDER_NAME,
        "attribution": ATTRIBUTION,
    }


def _normalise(query: str) -> str:
    return " ".join(query.split()).casefold()


def _wait_for_slot() -> None:
    """Block until the application-wide minimum request interval has passed."""
    elapsed = time.monotonic() - _last_request[0]
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_request[0] = time.monotonic()


def geocode(query: str, session: requests.Session | None = None) -> GeocodeResult:
    """Geocode one explicitly submitted public place name."""
    config = provider_config()
    if not config["enabled"]:
        return GeocodeResult(False, error="Place search is turned off for this deployment. Use map selection or coordinates.")
    cleaned = " ".join(str(query or "").split())
    if len(cleaned) < 3:
        return GeocodeResult(False, error="Enter at least three characters for a place name.")
    if len(cleaned) > MAX_QUERY_LENGTH:
        return GeocodeResult(False, error="That search is too long. Try a shorter public place name.")

    key = _normalise(cleaned)
    with _lock:
        if key in _memory_cache:
            _memory_cache.move_to_end(key)
            return _memory_cache[key]
        _wait_for_slot()
        try:
            http = session or requests
            response = http.get(
                str(config["url"]),
                params={
                    "q": cleaned,
                    "format": "jsonv2",
                    "limit": 5,
                    "countrycodes": "us",
                    "viewbox": STUDY_VIEWBOX,
                    "bounded": 0,
                },
                headers={"User-Agent": str(config["user_agent"]), "Accept-Language": "en"},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            return GeocodeResult(False, error="The place-search provider did not respond in time. Try again shortly, or use coordinates.")
        except requests.RequestException:
            return GeocodeResult(False, error="The place-search provider could not be reached. Use map selection or coordinates instead.")

        if response.status_code == 429:
            return GeocodeResult(False, error="The place-search provider is busy (rate limited). Please wait a moment before searching again.")
        if response.status_code != 200:
            return GeocodeResult(False, error=f"The place-search provider returned an error (HTTP {response.status_code}).")
        try:
            payload = response.json()
        except ValueError:
            return GeocodeResult(False, error="The place-search provider returned an unreadable response.")

        places = []
        for item in payload if isinstance(payload, list) else []:
            try:
                places.append(
                    Place(
                        name=str(item.get("display_name", "Unnamed place")),
                        lat=float(item["lat"]),
                        lon=float(item["lon"]),
                        kind=str(item.get("type", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        result = (
            GeocodeResult(True, places=places)
            if places
            else GeocodeResult(False, error="No matching public place was found. Check the spelling, add “Houston, TX”, or use coordinates.")
        )
        _memory_cache[key] = result
        while len(_memory_cache) > _CACHE_LIMIT:
            _memory_cache.popitem(last=False)
        return result


def clear_cache() -> None:
    with _lock:
        _memory_cache.clear()
