"""Ballparks as first-class entities.

The NBA repo keeps venue geography only as ``CITY_TO_LATLON`` and
``CITY_TO_TIMEZONE``, because travel distance and timezone shift are the whole
of what an arena contributes. Baseball is different: the park itself changes
the run environment far more than travel does, and the Stats API publishes
everything needed to model that on one endpoint -- outfield dimensions at
seven points, elevation, roof type, turf, and the field's compass orientation
(``azimuthAngle``), which is what turns a raw "12 mph, Out To RF" wind reading
into a physically meaningful quantity.

Park *factors* are deliberately **not** fetched. A published park factor is a
full-season snapshot, so joining it onto a mid-season game leaks the rest of
that season's results backwards. The park factor this project uses is derived
from its own team-game log with a strict as-of cut -- the same reasoning that
makes the NBA repo derive standings from its event log instead of scraping
snapshots.
"""

from __future__ import annotations

import pandas as pd

from mlb_pred.config.settings import SETTINGS
from mlb_pred.fetch_data.statsapi.client import statsapi_get
from mlb_pred.utils.general_utils import as_id

VENUE_COLUMNS = [
    "venue_id",
    "venue_name",
    "city",
    "state",
    "country",
    "latitude",
    "longitude",
    "elevation_ft",
    "azimuth_angle",
    "timezone_id",
    "timezone_offset",
    "capacity",
    "turf_type",
    "roof_type",
    "left_line",
    "left",
    "left_center",
    "center",
    "right_center",
    "right",
    "right_line",
    "active",
    "season",
]

VENUE_HYDRATE = "location,fieldInfo,timezone"


def fetch_venues(season: int | None = None) -> pd.DataFrame:
    """One row per MLB venue.

    ``season`` matters because dimensions and roof state are versioned: a
    park that moved its fences has a different ``fieldInfo`` per season, and
    joining a 2015 game onto 2026 dimensions would be quietly wrong.
    """
    payload = statsapi_get(
        "venues",
        {
            "sportId": SETTINGS.statsapi_sport_id,
            "season": season,
            "hydrate": VENUE_HYDRATE,
        },
    )

    rows = []
    for venue in payload.get("venues", []):
        location = venue.get("location") or {}
        coordinates = location.get("defaultCoordinates") or {}
        timezone = venue.get("timeZone") or {}
        field = venue.get("fieldInfo") or {}

        rows.append(
            {
                "venue_id": as_id(venue.get("id")),
                "venue_name": venue.get("name"),
                "city": location.get("city"),
                "state": location.get("stateAbbrev"),
                "country": location.get("country"),
                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),
                "elevation_ft": location.get("elevation"),
                # Compass bearing from home plate to centre field. Without it
                # a wind direction is uninterpretable across parks.
                "azimuth_angle": location.get("azimuthAngle"),
                "timezone_id": timezone.get("id"),
                "timezone_offset": timezone.get("offset"),
                "capacity": field.get("capacity"),
                "turf_type": field.get("turfType"),
                "roof_type": field.get("roofType"),
                "left_line": field.get("leftLine"),
                "left": field.get("left"),
                "left_center": field.get("leftCenter"),
                "center": field.get("center"),
                "right_center": field.get("rightCenter"),
                "right": field.get("right"),
                "right_line": field.get("rightLine"),
                "active": venue.get("active"),
                "season": season if season is not None else venue.get("season"),
            }
        )

    return pd.DataFrame(rows, columns=VENUE_COLUMNS)
