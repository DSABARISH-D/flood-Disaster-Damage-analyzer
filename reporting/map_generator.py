"""
Interactive Map Generator
==========================
Creates Folium maps with flood overlays, object markers,
and damage heatmaps for the dashboard.
"""

import numpy as np
import folium
from folium.plugins import HeatMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MapGenerator:
    """
    Generates interactive Folium maps for flood damage visualization.

    Maps include:
      - Base map with satellite/street tiles
      - Flood area polygon overlay
      - Object detection markers with popups
      - Damage intensity heatmap layer
    """

    def __init__(self):
        self.default_location = config.MAP_DEFAULT_LOCATION
        self.default_zoom = config.MAP_DEFAULT_ZOOM
        self.flood_color = config.MAP_FLOOD_COLOR
        self.marker_color = config.MAP_MARKER_COLOR

    def generate_map(
        self,
        location: dict = None,
        flood_mask: np.ndarray = None,
        detections: list = None,
        assessment_report: dict = None,
    ) -> folium.Map:
        """
        Generate a complete interactive map.

        Args:
            location: {"lat": float, "lon": float} or None
            flood_mask: Binary flood mask (H, W)
            detections: List of detection dicts from ObjectDetector
            assessment_report: Output from DamageAssessmentEngine

        Returns:
            folium.Map: Interactive map object
        """
        # Determine center location
        lat = location.get("lat", self.default_location[0]) if location else self.default_location[0]
        lon = location.get("lon", self.default_location[1]) if location else self.default_location[1]

        # Create base map
        m = folium.Map(
            location=[lat, lon],
            zoom_start=15 if location else self.default_zoom,
            tiles=None,
        )

        # Add tile layers
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Street Map",
        ).add_to(m)

        # Add flood overlay if mask provided
        if flood_mask is not None and location:
            self._add_flood_overlay(m, flood_mask, lat, lon)

        # Add detection markers
        if detections and location:
            self._add_detection_markers(m, detections, lat, lon)

        # Add heatmap layer
        if detections and location:
            self._add_heatmap(m, detections, lat, lon)

        # Add assessment info popup
        if assessment_report:
            self._add_info_popup(m, assessment_report, lat, lon)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m

    def _add_flood_overlay(self, m: folium.Map, flood_mask: np.ndarray, lat: float, lon: float):
        """Add a simplified flood polygon overlay to the map."""
        import cv2

        # Find contours in the flood mask
        contours, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        # Map image coordinates to geo coordinates
        # This is an approximation — in production you'd use proper georeferencing
        h, w = flood_mask.shape[:2]
        # Approximate span: ~0.005 degrees per 100 pixels
        deg_per_px_lat = 0.005 / 100
        deg_per_px_lon = 0.005 / 100

        flood_group = folium.FeatureGroup(name="Flood Zone")

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Skip tiny contours
                continue

            # Simplify contour for performance
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert pixel coords to lat/lon
            geo_coords = []
            for point in approx:
                px, py = point[0]
                point_lat = lat + (h / 2 - py) * deg_per_px_lat
                point_lon = lon + (px - w / 2) * deg_per_px_lon
                geo_coords.append([point_lat, point_lon])

            if len(geo_coords) >= 3:
                folium.Polygon(
                    locations=geo_coords,
                    color=self.flood_color,
                    fill=True,
                    fill_color=self.flood_color,
                    fill_opacity=0.4,
                    weight=2,
                    popup="Flood Zone",
                ).add_to(flood_group)

        flood_group.add_to(m)

    def _add_detection_markers(self, m: folium.Map, detections: list, lat: float, lon: float):
        """Add markers for each detected object."""
        marker_group = folium.FeatureGroup(name="Detected Objects")

        icon_map = {
            "person": ("user", "red"),
            "car": ("car", "blue"),
            "truck": ("truck", "darkblue"),
            "bus": ("bus", "purple"),
            "boat": ("ship", "orange"),
            "bicycle": ("bicycle", "green"),
            "motorcycle": ("motorcycle", "darkgreen"),
        }

        for i, det in enumerate(detections):
            cls = det["class"]
            icon_name, icon_color = icon_map.get(cls, ("info-sign", "gray"))

            # Offset markers slightly so they don't overlap
            marker_lat = lat + (np.random.random() - 0.5) * 0.002
            marker_lon = lon + (np.random.random() - 0.5) * 0.002

            # Status badge
            in_flood = det.get("in_flood_zone", False)
            status = "⚠️ IN FLOOD ZONE" if in_flood else "✅ Safe"

            popup_html = f"""
            <div style="font-family: Arial; min-width: 150px;">
                <h4 style="margin: 0; color: {'#e53935' if in_flood else '#43a047'};">
                    {cls.title()}
                </h4>
                <p style="margin: 4px 0;">
                    <b>Confidence:</b> {det['confidence']:.0%}<br>
                    <b>Status:</b> {status}<br>
                    <b>Detection #{i + 1}</b>
                </p>
            </div>
            """

            folium.Marker(
                location=[marker_lat, marker_lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(
                    color="red" if in_flood else "green",
                    icon=icon_name,
                    prefix="fa",
                ),
            ).add_to(marker_group)

        marker_group.add_to(m)

    def _add_heatmap(self, m: folium.Map, detections: list, lat: float, lon: float):
        """Add a damage intensity heatmap layer."""
        heat_data = []

        for det in detections:
            if det.get("in_flood_zone", False):
                # Objects in flood zone contribute to heatmap
                marker_lat = lat + (np.random.random() - 0.5) * 0.002
                marker_lon = lon + (np.random.random() - 0.5) * 0.002

                # Weight based on object type
                weight_map = {"person": 3.0, "bus": 2.5, "truck": 2.0, "car": 1.5, "boat": 1.0}
                weight = weight_map.get(det["class"], 1.0)

                heat_data.append([marker_lat, marker_lon, weight])

        if heat_data:
            HeatMap(
                heat_data,
                name="Damage Heatmap",
                radius=25,
                blur=15,
                max_zoom=18,
            ).add_to(m)

    def _add_info_popup(self, m: folium.Map, report: dict, lat: float, lon: float):
        """Add an informational popup at the center with assessment summary."""
        category = report.get("damage_category", "Unknown")
        severity = report.get("severity_score", 0)
        emoji = report.get("damage_category_emoji", "")
        flood_pct = report.get("flood_area_percentage", 0)

        color_map = {
            "Minor": "#ffc107",
            "Moderate": "#ff9800",
            "Severe": "#f44336",
            "Critical": "#212121",
        }
        bg_color = color_map.get(category, "#9e9e9e")

        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px; text-align: center;">
            <div style="background: {bg_color}; color: white; padding: 8px; border-radius: 6px 6px 0 0;">
                <h3 style="margin: 0;">{emoji} {category}</h3>
                <p style="margin: 2px 0;">Severity: {severity}/10</p>
            </div>
            <div style="padding: 8px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 6px 6px;">
                <p style="margin: 4px 0;"><b>Flood Coverage:</b> {flood_pct:.1f}%</p>
                <p style="margin: 4px 0;"><b>Objects Detected:</b> {report.get('total_objects_detected', 0)}</p>
                <p style="margin: 4px 0;"><b>In Flood Zone:</b> {report.get('objects_in_flood_zone', 0)}</p>
            </div>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="darkred", icon="exclamation-triangle", prefix="fa"),
        ).add_to(m)
