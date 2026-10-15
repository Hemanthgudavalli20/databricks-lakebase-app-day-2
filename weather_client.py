"""
Client for the National Weather Service public API (api.weather.gov).

No API key is required — the service is free and open. The only
requirement is a descriptive User-Agent header, which we set
automatically (overridable via the WEATHER_API_USER_AGENT env var).
"""

import os
from typing import Any

import requests

_BASE_URL = os.environ.get("WEATHER_API_BASE_URL", "https://api.weather.gov")
_DEFAULT_TIMEOUT = 30

# A descriptive User-Agent is required by the NWS API.
# Format: "app-name/version (contact-email)" per NWS guidelines.
_USER_AGENT = os.environ.get(
    "WEATHER_API_USER_AGENT",
    "databricks-lakebase-app/1.0 (contact@example.com)",
)


class WeatherClient:
    """Thin wrapper around the NWS API with a session + User-Agent."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def _fetch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Fetch and parse a JSON response from an NWS endpoint."""
        url = f"{self.base_url}{path}"
        resp = self._session.request("GET", url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_point_info(self, latitude: float, longitude: float, include_forecast: bool = False) -> dict:
        """
        Get metadata for a geographic point.

        Calls /points/{lat},{lon} and returns the full GeoJSON response
        whose ``properties`` dict includes the forecast office ID (``cwa``),
        grid coordinates (``gridId``, ``gridX``, ``gridY``), and URLs for
        forecasts and observing stations.

        When ``include_forecast`` is True, the response's ``properties``
        dict is also augmented with a ``_forecast`` key containing the full
        forecast GeoJSON (with ``properties.periods``).
        """
        point = self._fetch_json(f"/points/{latitude},{longitude}")
        if include_forecast:
            props = point.get("properties", {})
            forecast_url = props.get("forecast", "")
            if forecast_url:
                path = forecast_url.replace(self.base_url, "")
                props["_forecast"] = self._fetch_json(path)
        return point

    def get_location_name(self, latitude: float, longitude: float) -> dict:
        """
        Resolve a human-readable city and state for a geographic point.

        Reuses the /points/{lat},{lon} response (same call as
        get_point_info) and extracts the ``relativeLocation`` properties,
        which include the nearest city and state.

        Returns a dict with keys: ``city``, ``state`` (either may be
        ``None`` if the NWS API doesn't provide them, e.g. for ocean coords).
        """
        point = self.get_point_info(latitude, longitude)
        props = point.get("properties", {})
        rel = props.get("relativeLocation", {})
        rel_props = rel.get("properties", {})
        return {
            "city": rel_props.get("city"),
            "state": rel_props.get("state"),
        }

    def get_active_alerts(self, latitude: float, longitude: float) -> list[dict]:
        """
        Fetch active weather alerts (watches, warnings, advisories) for a
        specific point.

        Calls /alerts/active?point={lat},{lon} and returns the ``features``
        list from the GeoJSON FeatureCollection. Each feature's
        ``properties`` dict contains: ``event``, ``headline``, ``description``,
        ``severity``, ``urgency``, ``areaDesc``, ``effective``, ``onset``,
        ``expires``, ``ends``, ``senderName``, etc.
        """
        data = self._fetch_json(
            "/alerts/active", params={"point": f"{latitude},{longitude}"}
        )
        return data.get("features", [])

    def get_forecast_discussion(self, latitude: float, longitude: float) -> dict:
        """
        Fetch the latest Area Forecast Discussion for the office responsible
        for the given coordinates.

        Returns a dict with keys: ``office_id``, ``product_id``, ``issued``,
        ``product_text``. If none is available, ``product_id`` and
        ``product_text`` will be ``None``.
        """
        # Step 1: resolve the forecast office from the point metadata.
        point = self.get_point_info(latitude, longitude)
        props = point.get("properties", {})
        cwa = props.get("cwa")
        if not cwa:
            raise ValueError(
                f"Could not resolve forecast office for {latitude},{longitude}"
            )

        # Step 2: list recent AFD products for that office.
        afd_listing = self._fetch_json(f"/products/types/AFD/locations/{cwa}")
        products = afd_listing.get("@graph", [])
        if not products:
            return {
                "office_id": cwa,
                "product_id": None,
                "issued": None,
                "product_text": None,
            }

        latest = products[0]
        product_id = latest.get("id")

        # Step 3: fetch the full product text.
        product = self._fetch_json(f"/products/{product_id}")
        return {
            "office_id": cwa,
            "product_id": product_id,
            "issued": product.get("issuanceTime") or latest.get("issuanceTime"),
            "product_text": product.get("productText"),
        }
