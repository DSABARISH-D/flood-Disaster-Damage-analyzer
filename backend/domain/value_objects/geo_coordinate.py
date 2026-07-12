"""
Geo Coordinate Value Object
=============================
Represents a geographic location (latitude, longitude)
with validation and distance calculation.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeoCoordinate:
    """
    Immutable geographic coordinate.

    Attributes:
        lat: Latitude in decimal degrees (-90 to +90)
        lon: Longitude in decimal degrees (-180 to +180)
    """

    lat: float
    lon: float

    def __post_init__(self):
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"Latitude must be -90 to +90, got {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"Longitude must be -180 to +180, got {self.lon}")

    def distance_km(self, other: GeoCoordinate) -> float:
        """
        Calculate distance to another coordinate using the Haversine formula.

        Returns:
            Distance in kilometers.
        """
        R = 6371.0  # Earth's radius in km

        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {"lat": round(self.lat, 6), "lon": round(self.lon, 6)}

    def to_tuple(self) -> tuple[float, float]:
        """Return as (lat, lon) tuple for map libraries."""
        return (self.lat, self.lon)
