"""
weather_service.py

Server-side weather lookup for the Recommendation Engine.

Moving this call server-side (instead of the frontend's current
HeroSection.jsx, which calls OpenWeatherMap directly with a hardcoded key)
also fixes the previously-flagged exposed API key issue: the key now lives
in an environment variable on the server, never in the shipped frontend
bundle.

Setup required:
    export OPENWEATHER_API_KEY="your-key-here"

IMPORTANT: rotate the key that was previously hardcoded in HeroSection.jsx
before reusing it here -- treat it as already compromised, per the earlier
audit finding.
"""

import os
import requests

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Fallback used whenever we can't get a real reading -- keeps /analyze
# working even if weather is unavailable, misconfigured, or the person
# didn't grant location permission.
DEFAULT_WEATHER = {"temperature_c": 20.0, "condition": "mild"}


def _bucket_condition(temp_c, weather_main):
    """
    Collapse raw API data into the simple buckets the Recommendation
    Engine's scoring rules understand: hot / cold / rain / mild.
    """
    weather_main = (weather_main or "").lower()
    if "rain" in weather_main or "drizzle" in weather_main or "thunderstorm" in weather_main:
        return "rain"
    if temp_c >= 28:
        return "hot"
    if temp_c <= 12:
        return "cold"
    return "mild"


def get_current_weather(lat, lon):
    """
    Returns: {"temperature_c": float, "condition": str}

    Never raises -- any failure (missing key, no network, bad coordinates)
    falls back to DEFAULT_WEATHER so a weather outage never breaks /analyze.
    """
    if not OPENWEATHER_API_KEY or lat is None or lon is None:
        return dict(DEFAULT_WEATHER)

    try:
        resp = requests.get(
            OPENWEATHER_URL,
            params={
                "lat": lat,
                "lon": lon,
                "units": "metric",
                "appid": OPENWEATHER_API_KEY,
            },
            timeout=5,
        )
        data = resp.json()
        temp_c = data.get("main", {}).get("temp")
        weather_main = (data.get("weather") or [{}])[0].get("main")

        if temp_c is None:
            return dict(DEFAULT_WEATHER)

        return {
            "temperature_c": temp_c,
            "condition": _bucket_condition(temp_c, weather_main),
        }
    except Exception as e:
        print(f"[weather_service] lookup failed, using default: {e}")
        return dict(DEFAULT_WEATHER)
