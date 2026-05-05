"""
sensor_ingestor.py
Sensor event data models and ingestion pipeline.

In production this connects to Azure IoT Hub / Event Hub.
For the demo it simulates events from a local queue.

SensorEvent lifecycle:
  raw_event → parse → validate → enrich → publish to classifier
"""

from __future__ import annotations

import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO     = "INFO"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    # HVAC
    TEMPERATURE_HIGH        = "temperature_high"
    TEMPERATURE_LOW         = "temperature_low"
    COOLING_FAILURE         = "cooling_failure"
    HEATING_FAILURE         = "heating_failure"
    THERMOSTAT_FAULT        = "thermostat_fault"
    SETPOINT_DEVIATION      = "setpoint_deviation"
    CO2_HIGH                = "co2_high"
    IAQ_ALERT               = "iaq_alert"
    FILTER_DIRTY            = "filter_dirty"
    AIRFLOW_LOW             = "airflow_low"
    REFRIGERANT_LEAK        = "refrigerant_leak"
    COMPRESSOR_FAULT        = "compressor_fault"
    # Electrical
    POWER_FAILURE           = "power_failure"
    VOLTAGE_DROP            = "voltage_drop"
    BREAKER_TRIP            = "breaker_trip"
    LIGHTING_FAILURE        = "lighting_failure"
    GENERATOR_FAULT         = "generator_fault"
    UPS_FAULT               = "ups_fault"
    BATTERY_LOW             = "battery_low"
    # Plumbing
    WATER_LEAK              = "water_leak"
    FLOOD_DETECTION         = "flood_detection"
    HOT_WATER_FAILURE       = "hot_water_failure"
    DRAINAGE_BLOCKAGE       = "drainage_blockage"
    PUMP_FAULT              = "pump_fault"
    # Refrigeration
    FRIDGE_TEMP_HIGH        = "fridge_temp_high"
    FREEZER_TEMP_HIGH       = "freezer_temp_high"
    DEFROST_FAULT           = "defrost_fault"
    # Safety
    FIRE_ALARM              = "fire_alarm"
    SMOKE_DETECTION         = "smoke_detection"
    # Security
    DOOR_FORCED             = "door_forced"
    CAMERA_OFFLINE          = "camera_offline"
    # Mechanical
    VIBRATION_HIGH          = "vibration_high"
    BEARING_FAULT           = "bearing_fault"
    PUMP_FAILURE            = "pump_failure"
    # BAS
    CONTROLLER_OFFLINE      = "controller_offline"
    COMMUNICATION_FAULT     = "communication_fault"
    # Energy
    ENERGY_SPIKE            = "energy_spike"
    HIGH_CONSUMPTION        = "high_consumption"


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """Raw reading from a sensor."""
    parameter: str          # e.g. "temperature", "humidity", "voltage"
    value: float
    unit: str               # e.g. "°F", "%RH", "V"
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None

    @property
    def is_out_of_range(self) -> bool:
        if self.threshold_max is not None and self.value > self.threshold_max:
            return True
        if self.threshold_min is not None and self.value < self.threshold_min:
            return True
        return False

    @property
    def deviation_pct(self) -> float:
        """How far out of range as a percentage of the threshold."""
        if self.threshold_max and self.value > self.threshold_max:
            return round((self.value - self.threshold_max) / self.threshold_max * 100, 2)
        if self.threshold_min and self.value < self.threshold_min:
            return round((self.threshold_min - self.value) / self.threshold_min * 100, 2)
        return 0.0


@dataclass
class SensorEvent:
    """
    Canonical sensor event after parsing/enrichment.
    Maps 1:1 to a single alert from a physical asset sensor.
    """
    event_id: str                   = field(default_factory=lambda: str(uuid.uuid4()))
    sensor_id: str                  = ""
    asset_id: str                   = ""      # SFM asset identifier
    asset_name: str                 = ""
    asset_type: str                 = ""      # e.g. "AHU", "Chiller"
    alert_type: str                 = ""      # AlertType value
    severity: str                   = AlertSeverity.MEDIUM
    reading: Optional[SensorReading]= None
    location_id: str                = ""
    building: str                   = ""
    floor: str                      = ""
    room: str                       = ""
    timestamp: str                  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_payload: Dict[str, Any]     = field(default_factory=dict)
    enriched: bool                  = False
    sfm_record: Optional[dict]      = None    # populated after SFM lookup

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.reading:
            d["reading"] = asdict(self.reading)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SensorEvent":
        reading_data = data.pop("reading", None)
        event = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if reading_data:
            event.reading = SensorReading(**reading_data)
        return event


# ── Ingestor ──────────────────────────────────────────────────────────────────

class SensorIngestor:
    """
    Receives raw sensor payloads (webhook, MQTT, IoT Hub),
    parses them into SensorEvent objects, and passes them
    to registered handlers.

    Production: swap _read_iot_hub() with Azure IoT Hub SDK.
    """

    def __init__(self):
        self._handlers: List[Callable[[SensorEvent], None]] = []
        self._event_log: List[SensorEvent] = []    # in-memory log (use DB in prod)

    def register_handler(self, fn: Callable[[SensorEvent], None]):
        """Register a callback that receives enriched SensorEvents."""
        self._handlers.append(fn)

    def ingest(self, raw: dict) -> SensorEvent:
        """
        Parse a raw sensor payload and push to handlers.

        Expected raw keys (flexible — we try multiple field name variants):
          sensor_id, asset_id, asset_name, asset_type,
          alert_type, severity, reading{param,value,unit,min,max},
          location_id, building, floor, room, timestamp
        """
        event = self._parse(raw)
        event = self._enrich(event)
        self._event_log.append(event)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error("Handler %s failed: %s", handler.__name__, exc)
        return event

    def ingest_batch(self, events: List[dict]) -> List[SensorEvent]:
        return [self.ingest(e) for e in events]

    def get_recent_events(self, limit: int = 100) -> List[SensorEvent]:
        return self._event_log[-limit:]

    # ── private ────────────────────────────────────────────────────────────────

    def _parse(self, raw: dict) -> SensorEvent:
        reading = None
        r = raw.get("reading") or raw.get("sensor_reading") or {}
        if r:
            reading = SensorReading(
                parameter=r.get("parameter", r.get("param", "unknown")),
                value=float(r.get("value", 0)),
                unit=r.get("unit", ""),
                threshold_min=r.get("threshold_min") or r.get("min"),
                threshold_max=r.get("threshold_max") or r.get("max"),
            )

        return SensorEvent(
            event_id=raw.get("event_id", str(uuid.uuid4())),
            sensor_id=str(raw.get("sensor_id", "")),
            asset_id=str(raw.get("asset_id", raw.get("sfm_asset_id", ""))),
            asset_name=str(raw.get("asset_name", "")),
            asset_type=str(raw.get("asset_type", raw.get("equipment_type", ""))),
            alert_type=str(raw.get("alert_type", raw.get("alert", ""))).lower(),
            severity=str(raw.get("severity", AlertSeverity.MEDIUM)),
            reading=reading,
            location_id=str(raw.get("location_id", "")),
            building=str(raw.get("building", raw.get("building_name", ""))),
            floor=str(raw.get("floor", "")),
            room=str(raw.get("room", "")),
            timestamp=str(raw.get("timestamp", datetime.now(timezone.utc).isoformat())),
            raw_payload=raw,
        )

    def _enrich(self, event: SensorEvent) -> SensorEvent:
        """Add derived fields. In production: query asset DB, resolve IDs etc."""
        event.enriched = True
        # Normalise severity based on alert type
        critical_alerts = {
            AlertType.FIRE_ALARM, AlertType.SMOKE_DETECTION,
            AlertType.POWER_FAILURE, AlertType.WATER_LEAK,
            AlertType.FLOOD_DETECTION,
        }
        high_alerts = {
            AlertType.COOLING_FAILURE, AlertType.HEATING_FAILURE,
            AlertType.GENERATOR_FAULT, AlertType.UPS_FAULT,
            AlertType.DRAINAGE_BLOCKAGE,
        }
        if event.alert_type in {a.value for a in critical_alerts}:
            event.severity = AlertSeverity.CRITICAL
        elif event.alert_type in {a.value for a in high_alerts}:
            event.severity = AlertSeverity.HIGH
        return event


# ── Convenience factory ───────────────────────────────────────────────────────

def make_demo_events() -> List[dict]:
    """
    Returns a list of realistic demo sensor event payloads
    covering all severity levels and alert types for demo/testing.
    """
    return [
        {
            "sensor_id": "SNS-HVAC-001", "asset_id": "SFM-AHU-B1-01",
            "asset_name": "AHU-1 Level B1", "asset_type": "AHU",
            "alert_type": "temperature_high", "severity": "HIGH",
            "reading": {"parameter": "supply_air_temp", "value": 78.5, "unit": "°F",
                        "threshold_min": 55, "threshold_max": 72},
            "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
            "building": "HQ Building A", "floor": "B1", "room": "Mechanical Room 1",
            "timestamp": "2025-05-04T08:00:00Z",
        },
        {
            "sensor_id": "SNS-ELEC-002", "asset_id": "SFM-PANEL-FL3-02",
            "asset_name": "Electrical Panel FL3", "asset_type": "Electrical Panel",
            "alert_type": "breaker_trip", "severity": "CRITICAL",
            "reading": {"parameter": "current", "value": 95.0, "unit": "A",
                        "threshold_max": 80},
            "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
            "building": "HQ Building A", "floor": "3", "room": "Electrical Room",
            "timestamp": "2025-05-04T09:15:00Z",
        },
        {
            "sensor_id": "SNS-REFRIG-003", "asset_id": "SFM-FREEZER-CAFE-01",
            "asset_name": "Walk-in Freezer Cafeteria", "asset_type": "Freezer",
            "alert_type": "freezer_temp_high", "severity": "HIGH",
            "reading": {"parameter": "temperature", "value": 12.0, "unit": "°F",
                        "threshold_max": 0},
            "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
            "building": "HQ Building A", "floor": "1", "room": "Cafeteria",
            "timestamp": "2025-05-04T10:30:00Z",
        },
        {
            "sensor_id": "SNS-THERM-004", "asset_id": "SFM-THERM-FL2-01",
            "asset_name": "Thermostat Zone 2A", "asset_type": "Thermostat",
            "alert_type": "setpoint_deviation", "severity": "MEDIUM",
            "reading": {"parameter": "room_temp", "value": 85.2, "unit": "°F",
                        "threshold_min": 68, "threshold_max": 76},
            "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
            "building": "HQ Building A", "floor": "2", "room": "Open Office 2A",
            "timestamp": "2025-05-04T11:00:00Z",
        },
        {
            "sensor_id": "SNS-PUMP-005", "asset_id": "SFM-PUMP-CWR-01",
            "asset_name": "Chilled Water Pump 1", "asset_type": "Pump",
            "alert_type": "vibration_high", "severity": "MEDIUM",
            "reading": {"parameter": "vibration", "value": 12.4, "unit": "mm/s",
                        "threshold_max": 7.1},
            "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
            "building": "HQ Building A", "floor": "B1", "room": "Plant Room",
            "timestamp": "2025-05-04T12:00:00Z",
        },
    ]
