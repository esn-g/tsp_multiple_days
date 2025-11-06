"""Simple helper utilities for talking to the Mapbox geocoding API.

This module now keeps a lightweight on-disk cache of successful geocoding
lookups.  The cache prevents the application from repeatedly hitting the API
for addresses that have already been resolved which both speeds up the
execution and avoids exhausting the API quota during development.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Tuple

import requests

MAPBOX_TOKEN = "pk.eyJ1Ijoib3B0LWRldiIsImEiOiJjbWhsMDI4aWoxbW55MmpxcXgycDR3a2xyIn0.4wATU3zYEWQIIxnwl6w0oQ"

# ---------------------------------------------------------------------------
# Cache management helpers
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).with_name("geocode_cache.json")
_CACHE_LOCK = RLock()
_CACHE: Optional[Dict[str, Tuple[float, float]]] = None


def _normalise_address(address: str) -> str:
    """Return a canonical representation of an address for caching purposes."""

    return " ".join(str(address).strip().lower().split())


def _load_cache() -> Dict[str, Tuple[float, float]]:
    """Load the geocoding cache from disk, returning an in-memory dictionary."""

    global _CACHE
    if _CACHE is None:
        if _CACHE_PATH.exists():
            try:
                with _CACHE_PATH.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    # Ensure values are stored as tuples for immutability.
                    _CACHE = {key: tuple(value) for key, value in data.items()}
            except (json.JSONDecodeError, OSError):
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def _save_cache(cache: Dict[str, Tuple[float, float]]) -> None:
    """Persist the cache to disk."""

    tmp_path = _CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    tmp_path.replace(_CACHE_PATH)


def clear_cache() -> None:
    """Remove the cached addresses from both memory and disk."""

    global _CACHE
    with _CACHE_LOCK:
        _CACHE = {}
        if _CACHE_PATH.exists():
            try:
                _CACHE_PATH.unlink()
            except OSError:
                pass


def forward_geocode(address: str, *, use_cache: bool = True, force_refresh: bool = False) -> Optional[Tuple[float, float]]:
    """Return the (longitude, latitude) pair for ``address``.

    Parameters
    ----------
    address:
        Human readable address to forward geocode.
    use_cache:
        When ``True`` (default) a memoised value is returned if available.
    force_refresh:
        Ignore cached results and fetch a fresh value from the API.
    """

    cache_key = _normalise_address(address)
    if use_cache and not force_refresh:
        with _CACHE_LOCK:
            cache = _load_cache()
            if cache_key in cache:
                return cache[cache_key]

    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "limit": 1,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    coordinates: Optional[Tuple[float, float]] = None
    if data.get("features"):
        geometry = data["features"][0].get("geometry", {})
        coords = geometry.get("coordinates")
        if isinstance(coords, list) and len(coords) == 2:
            coordinates = (float(coords[0]), float(coords[1]))

    if coordinates and use_cache:
        with _CACHE_LOCK:
            cache = _load_cache()
            cache[cache_key] = coordinates
            _save_cache(cache)

    return coordinates

def reverse_geocode(lon: float, lat: float) -> Optional[str]:
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    features = data.get("features") or []
    if features:
        return features[0].get("place_name")
    return None

def main():
    # Example usage
    coords = forward_geocode("Villa Horn 1, 72592 Västerås, Sweden")
    print("Coordinates:", coords)

    address = reverse_geocode(coords[0], coords[1])
    print("Address:", address)
    
if __name__== "__main__":
    main()
    
    
    
    

