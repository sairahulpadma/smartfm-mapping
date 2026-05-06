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
    Comprehensive test dataset: 35 sensor events covering every service
    classification (23) and all 4 pipeline tiers.

    Tier breakdown:
      Tier 1 – Perfect Match (≥85%)  → AUTO_CREATE   (events 001–023)
      Tier 2 – Partial Match (50–84%) → REVIEW queue  (events 024–028)
      Tier 3 – LLM Reasoned  (30–49%) → REVIEW queue  (events 029–032)
      Tier 4 – No Match      (<30%)   → NO_ACTION     (events 033–035)

    Service classifications covered (one Perfect per classification):
      HVAC-COOL, HVAC-HEAT, HVAC-CTRL, HVAC-VENT, HVAC-FILT, HVAC-ZONE,
      ELEC-PWR, ELEC-LIGHT, ELEC-EMERG, ELEC-CRIT,
      PLMB-LEAK, PLMB-HW, PLMB-DRN,
      REFRIG, BMS, FIRE, ELEV, MECH-VIB, MECH-PUMP,
      SEC-AC, SEC-CCTV, ENERGY, COOL-TWR
    """
    _LOC = "f754334d-17cc-4890-bc58-2a4e1a386549"

    return [
        # ══════════════════════════════════════════════════════════════════════
        # TIER 1 — PERFECT MATCH  (asset_type + alert_type exact catalog match)
        # Expected: Perfect Match → AUTO_CREATE
        # ══════════════════════════════════════════════════════════════════════

        # 001 — HVAC Cooling System Failure  (HIGH)
        {
            "sensor_id": "SNS-001", "asset_id": "SFM-AHU-B1-01",
            "asset_name": "AHU-1 Level B1", "asset_type": "AHU",
            "alert_type": "temperature_high", "severity": "HIGH",
            "reading": {"parameter": "supply_air_temp", "value": 78.5, "unit": "°F",
                        "threshold_min": 55, "threshold_max": 72},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Mechanical Room 1", "timestamp": "2026-05-06T06:00:00Z",
        },
        # 002 — HVAC Heating System Failure  (HIGH)
        {
            "sensor_id": "SNS-002", "asset_id": "SFM-BOILER-B2-01",
            "asset_name": "Gas Boiler B2", "asset_type": "Boiler",
            "alert_type": "heating_failure", "severity": "HIGH",
            "reading": {"parameter": "supply_water_temp", "value": 98.0, "unit": "°F",
                        "threshold_min": 140, "threshold_max": 200},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B2",
            "room": "Boiler Room", "timestamp": "2026-05-06T06:10:00Z",
        },
        # 003 — HVAC Thermostat / Controls Issue  (MEDIUM)
        {
            "sensor_id": "SNS-003", "asset_id": "SFM-THERM-FL2-01",
            "asset_name": "Thermostat Zone 2A", "asset_type": "Thermostat",
            "alert_type": "setpoint_deviation", "severity": "MEDIUM",
            "reading": {"parameter": "room_temp", "value": 85.2, "unit": "°F",
                        "threshold_min": 68, "threshold_max": 76},
            "location_id": _LOC, "building": "HQ Building A", "floor": "2",
            "room": "Open Office 2A", "timestamp": "2026-05-06T06:20:00Z",
        },
        # 004 — HVAC Ventilation / Air Quality  (MEDIUM)
        {
            "sensor_id": "SNS-004", "asset_id": "SFM-ERU-RF-01",
            "asset_name": "Energy Recovery Unit Roof", "asset_type": "ERU",
            "alert_type": "co2_high", "severity": "MEDIUM",
            "reading": {"parameter": "co2_ppm", "value": 1250.0, "unit": "ppm",
                        "threshold_max": 1000},
            "location_id": _LOC, "building": "HQ Building A", "floor": "RF",
            "room": "Penthouse Plant Room", "timestamp": "2026-05-06T06:30:00Z",
        },
        # 005 — Electrical Power Failure / Outage  (CRITICAL)
        {
            "sensor_id": "SNS-005", "asset_id": "SFM-PANEL-FL3-02",
            "asset_name": "Electrical Panel FL3", "asset_type": "Electrical Panel",
            "alert_type": "breaker_trip", "severity": "CRITICAL",
            "reading": {"parameter": "current", "value": 95.0, "unit": "A",
                        "threshold_max": 80},
            "location_id": _LOC, "building": "HQ Building A", "floor": "3",
            "room": "Electrical Room 3E", "timestamp": "2026-05-06T06:40:00Z",
        },
        # 006 — Electrical Lighting Issue  (LOW)
        {
            "sensor_id": "SNS-006", "asset_id": "SFM-LIGHT-PANEL-FL4",
            "asset_name": "Lighting Panel Floor 4 East", "asset_type": "Lighting Panel",
            "alert_type": "lighting_failure", "severity": "LOW",
            "reading": {"parameter": "circuit_status", "value": 0.0, "unit": "on/off",
                        "threshold_min": 1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "4",
            "room": "East Corridor", "timestamp": "2026-05-06T06:50:00Z",
        },
        # 007 — Electrical Emergency Power System  (CRITICAL)
        {
            "sensor_id": "SNS-007", "asset_id": "SFM-GEN-MAIN-01",
            "asset_name": "Emergency Generator Main", "asset_type": "Generator",
            "alert_type": "generator_fault", "severity": "CRITICAL",
            "reading": {"parameter": "fuel_level", "value": 12.0, "unit": "%",
                        "threshold_min": 25},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B2",
            "room": "Generator Room", "timestamp": "2026-05-06T07:00:00Z",
        },
        # 008 — Plumbing Water Leak Detection  (CRITICAL)
        {
            "sensor_id": "SNS-008", "asset_id": "SFM-WL-SENSOR-B1-01",
            "asset_name": "Water Leak Sensor B1 IT Room", "asset_type": "Water Leak Sensor",
            "alert_type": "water_leak", "severity": "CRITICAL",
            "reading": {"parameter": "moisture", "value": 1.0, "unit": "wet/dry",
                        "threshold_max": 0},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "IT Server Room", "timestamp": "2026-05-06T07:10:00Z",
        },
        # 009 — Plumbing Domestic Hot Water Issue  (MEDIUM)
        {
            "sensor_id": "SNS-009", "asset_id": "SFM-HWH-B1-01",
            "asset_name": "Domestic Hot Water Heater B1", "asset_type": "Water Heater",
            "alert_type": "hot_water_failure", "severity": "MEDIUM",
            "reading": {"parameter": "outlet_temp", "value": 85.0, "unit": "°F",
                        "threshold_min": 120},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Plant Room", "timestamp": "2026-05-06T07:20:00Z",
        },
        # 010 — Plumbing Sanitary / Drainage Issue  (HIGH)
        {
            "sensor_id": "SNS-010", "asset_id": "SFM-SEWPMP-B2-01",
            "asset_name": "Sewage Ejector Pump B2", "asset_type": "Sewage Pump",
            "alert_type": "drainage_blockage", "severity": "HIGH",
            "reading": {"parameter": "sump_level", "value": 95.0, "unit": "%",
                        "threshold_max": 80},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B2",
            "room": "Sump Pit B2", "timestamp": "2026-05-06T07:30:00Z",
        },
        # 011 — Refrigeration Temperature Variance  (HIGH)
        {
            "sensor_id": "SNS-011", "asset_id": "SFM-FREEZER-CAFE-01",
            "asset_name": "Walk-in Freezer Cafeteria", "asset_type": "Freezer",
            "alert_type": "freezer_temp_high", "severity": "HIGH",
            "reading": {"parameter": "temperature", "value": 12.0, "unit": "°F",
                        "threshold_max": 0},
            "location_id": _LOC, "building": "HQ Building A", "floor": "1",
            "room": "Cafeteria Storage", "timestamp": "2026-05-06T07:40:00Z",
        },
        # 012 — BMS/BAS Control System Fault  (HIGH)
        {
            "sensor_id": "SNS-012", "asset_id": "SFM-BAS-CTRL-01",
            "asset_name": "BAS Central Controller", "asset_type": "BAS",
            "alert_type": "controller_offline", "severity": "HIGH",
            "reading": {"parameter": "heartbeat", "value": 0.0, "unit": "online/offline",
                        "threshold_min": 1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "BAS Room", "timestamp": "2026-05-06T07:50:00Z",
        },
        # 013 — Fire & Life Safety Alarm  (CRITICAL)
        {
            "sensor_id": "SNS-013", "asset_id": "SFM-FIRE-PANEL-01",
            "asset_name": "Fire Alarm Panel Main Lobby", "asset_type": "Fire Alarm Panel",
            "alert_type": "fire_alarm", "severity": "CRITICAL",
            "reading": {"parameter": "alarm_state", "value": 1.0, "unit": "triggered",
                        "threshold_max": 0},
            "location_id": _LOC, "building": "HQ Building A", "floor": "1",
            "room": "Main Lobby", "timestamp": "2026-05-06T08:00:00Z",
        },
        # 014 — Elevator / Vertical Transport Fault  (CRITICAL)
        {
            "sensor_id": "SNS-014", "asset_id": "SFM-ELEV-MAIN-02",
            "asset_name": "Passenger Elevator 2", "asset_type": "Elevator",
            "alert_type": "elevator_fault", "severity": "CRITICAL",
            "reading": {"parameter": "door_status", "value": 0.0, "unit": "open/closed",
                        "threshold_min": 1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "3",
            "room": "Elevator Shaft 2", "timestamp": "2026-05-06T08:10:00Z",
        },
        # 015 — Mechanical Vibration / Noise Anomaly  (MEDIUM)
        {
            "sensor_id": "SNS-015", "asset_id": "SFM-PUMP-CWR-01",
            "asset_name": "Chilled Water Pump CWP-1", "asset_type": "Pump",
            "alert_type": "vibration_high", "severity": "MEDIUM",
            "reading": {"parameter": "vibration_rms", "value": 12.4, "unit": "mm/s",
                        "threshold_max": 7.1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Central Plant", "timestamp": "2026-05-06T08:20:00Z",
        },
        # 016 — Mechanical Pump Failure  (HIGH)
        {
            "sensor_id": "SNS-016", "asset_id": "SFM-CWP-002",
            "asset_name": "Chilled Water Pump CWP-2", "asset_type": "Chilled Water Pump",
            "alert_type": "pump_failure", "severity": "HIGH",
            "reading": {"parameter": "differential_pressure", "value": 2.1, "unit": "psi",
                        "threshold_min": 8.0},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Central Plant", "timestamp": "2026-05-06T08:30:00Z",
        },
        # 017 — Access Control Door / Lock Fault  (HIGH)
        {
            "sensor_id": "SNS-017", "asset_id": "SFM-ACP-DOOR-5W",
            "asset_name": "Access Control Panel West Wing", "asset_type": "Access Control Panel",
            "alert_type": "door_forced", "severity": "HIGH",
            "reading": {"parameter": "door_tamper", "value": 1.0, "unit": "triggered",
                        "threshold_max": 0},
            "location_id": _LOC, "building": "HQ Building A", "floor": "5",
            "room": "West Wing Entrance", "timestamp": "2026-05-06T08:40:00Z",
        },
        # 018 — CCTV Surveillance Camera Fault  (MEDIUM)
        {
            "sensor_id": "SNS-018", "asset_id": "SFM-CCTV-PARK-03",
            "asset_name": "CCTV Camera Parking Lot 3", "asset_type": "CCTV Camera",
            "alert_type": "camera_offline", "severity": "MEDIUM",
            "reading": {"parameter": "video_signal", "value": 0.0, "unit": "online/offline",
                        "threshold_min": 1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "P1",
            "room": "Parking Level 1", "timestamp": "2026-05-06T08:50:00Z",
        },
        # 019 — Energy Management High Consumption  (LOW)
        {
            "sensor_id": "SNS-019", "asset_id": "SFM-EMETER-MAIN-01",
            "asset_name": "Energy Meter Main Distribution", "asset_type": "Energy Meter",
            "alert_type": "energy_spike", "severity": "LOW",
            "reading": {"parameter": "demand_kw", "value": 485.0, "unit": "kW",
                        "threshold_max": 400},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Main Electrical Room", "timestamp": "2026-05-06T09:00:00Z",
        },
        # 020 — Cooling Tower Maintenance Required  (MEDIUM)
        {
            "sensor_id": "SNS-020", "asset_id": "SFM-CT-RF-01",
            "asset_name": "Cooling Tower CT-1 Roof", "asset_type": "Cooling Tower",
            "alert_type": "water_level_low", "severity": "MEDIUM",
            "reading": {"parameter": "basin_level", "value": 18.0, "unit": "%",
                        "threshold_min": 30},
            "location_id": _LOC, "building": "HQ Building A", "floor": "RF",
            "room": "Roof Plant Area", "timestamp": "2026-05-06T09:10:00Z",
        },
        # 021 — HVAC Filter Change / Air Handler Maintenance  [MULTI-ASSET: LOW]
        {
            "sensor_id": "SNS-021", "asset_id": "SFM-FCU-FL6-08",
            "asset_name": "Fan Coil Unit Floor 6 Zone B", "asset_type": "FCU",
            "alert_type": "filter_dirty", "severity": "LOW",
            "reading": {"parameter": "static_pressure", "value": 1.8, "unit": "in-wg",
                        "threshold_max": 1.2},
            "location_id": _LOC, "building": "HQ Building A", "floor": "6",
            "room": "Zone B Open Plan", "timestamp": "2026-05-06T09:20:00Z",
        },
        # 022 — HVAC Zone System Total Failure  [MULTI-ASSET: CRITICAL]
        {
            "sensor_id": "SNS-022", "asset_id": "SFM-DDC-ZONE4-01",
            "asset_name": "DDC Zone Controller Floor 4", "asset_type": "VAV Controller",
            "alert_type": "zone_failure", "severity": "CRITICAL",
            "reading": {"parameter": "zone_status", "value": 0.0, "unit": "active/fault",
                        "threshold_min": 1},
            "location_id": _LOC, "building": "HQ Building A", "floor": "4",
            "room": "Zone Controller Panel", "timestamp": "2026-05-06T09:30:00Z",
        },
        # 023 — Electrical Critical Power Infrastructure  [MULTI-ASSET: CRITICAL]
        {
            "sensor_id": "SNS-023", "asset_id": "SFM-UPS-SERVER-01",
            "asset_name": "UPS Server Room Critical Load", "asset_type": "UPS",
            "alert_type": "ups_fault", "severity": "CRITICAL",
            "reading": {"parameter": "battery_runtime", "value": 4.0, "unit": "min",
                        "threshold_min": 15},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Server Room", "timestamp": "2026-05-06T09:40:00Z",
        },

        # ══════════════════════════════════════════════════════════════════════
        # TIER 2 — PARTIAL MATCH  (asset_type or alert_type is fuzzy / adjacent)
        # Expected: Partial Match → REVIEW queue
        # ══════════════════════════════════════════════════════════════════════

        # 024 — Chiller with non-catalog alert (low_flow_fault) → Partial HVAC Cooling
        {
            "sensor_id": "SNS-024", "asset_id": "SFM-CHLR-ROOF-01",
            "asset_name": "Centrifugal Chiller Roof Plant", "asset_type": "Chiller",
            "alert_type": "low_flow_fault", "severity": "HIGH",
            "reading": {"parameter": "chilled_water_flow", "value": 180.0, "unit": "GPM",
                        "threshold_min": 350},
            "location_id": _LOC, "building": "Tower B", "floor": "RF",
            "room": "Roof Plant Room", "timestamp": "2026-05-06T09:50:00Z",
        },
        # 025 — ERU with temperature_low → borderline Heating vs Ventilation
        {
            "sensor_id": "SNS-025", "asset_id": "SFM-ERU-FL1-01",
            "asset_name": "Energy Recovery Unit Level 1", "asset_type": "ERU",
            "alert_type": "temperature_low", "severity": "MEDIUM",
            "reading": {"parameter": "supply_air_temp", "value": 52.0, "unit": "°F",
                        "threshold_min": 60},
            "location_id": _LOC, "building": "Tower B", "floor": "1",
            "room": "Mechanical Mezzanine", "timestamp": "2026-05-06T10:00:00Z",
        },
        # 026 — DDC Panel with sensor_offline (not in catalog alert_types) → Partial BMS
        {
            "sensor_id": "SNS-026", "asset_id": "SFM-DDC-FL2-03",
            "asset_name": "DDC Panel Floor 2 North", "asset_type": "DDC Panel",
            "alert_type": "sensor_offline", "severity": "MEDIUM",
            "reading": {"parameter": "point_count_online", "value": 6.0, "unit": "points",
                        "threshold_min": 24},
            "location_id": _LOC, "building": "Tower B", "floor": "2",
            "room": "Controls Closet 2N", "timestamp": "2026-05-06T10:10:00Z",
        },
        # 027 — Condenser Pump + overtemperature_alarm (not in pump catalog alerts)
        #        asset_score=100% (Condenser Pump ∈ Mechanical-Pump catalog)
        #        alert_score≈30% (character similarity with temperature_high) → composite≈57% → Partial
        {
            "sensor_id": "SNS-027", "asset_id": "SFM-CDP-B1-02",
            "asset_name": "Condenser Water Pump CDP-2", "asset_type": "Condenser Pump",
            "alert_type": "overtemperature_alarm", "severity": "HIGH",
            "reading": {"parameter": "motor_temp", "value": 112.0, "unit": "°C",
                        "threshold_max": 85},
            "location_id": _LOC, "building": "Tower B", "floor": "B1",
            "room": "Condenser Plant", "timestamp": "2026-05-06T10:20:00Z",
        },
        # 028 — NVR Storage Server + disk_health_warning (not in CCTV catalog alerts)
        #        asset_score=100% (NVR ∈ CCTV/Surveillance catalog)
        #        alert_score≈30% (character similarity with storage_full) → composite≈57% → Partial
        {
            "sensor_id": "SNS-028", "asset_id": "SFM-NVR-PARK-01",
            "asset_name": "NVR Parking Structure Recorder", "asset_type": "NVR",
            "alert_type": "disk_health_warning", "severity": "MEDIUM",
            "reading": {"parameter": "disk_health_score", "value": 28.0, "unit": "%",
                        "threshold_min": 70},
            "location_id": _LOC, "building": "Parking Structure", "floor": "P1",
            "room": "NVR Cabinet", "timestamp": "2026-05-06T10:30:00Z",
        },

        # ══════════════════════════════════════════════════════════════════════
        # TIER 3 — LLM REASONED  (ambiguous type + alert → needs GPT-5.5)
        # Expected: LLM Reasoned → REVIEW queue
        # ══════════════════════════════════════════════════════════════════════

        # 029 — Building Control Unit + network_loss → LLM → BMS/BAS
        {
            "sensor_id": "SNS-029", "asset_id": "SFM-BCU-HQ-01",
            "asset_name": "Building Control Unit HQ", "asset_type": "Building Control Unit",
            "alert_type": "network_loss", "severity": "HIGH",
            "reading": {"parameter": "packet_loss_pct", "value": 78.0, "unit": "%",
                        "threshold_max": 5},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B1",
            "room": "Network Comms Room", "timestamp": "2026-05-06T10:40:00Z",
        },
        # 030 — Hydronic Manifold + pressure_deviation → LLM → Mechanical/Plumbing
        {
            "sensor_id": "SNS-030", "asset_id": "SFM-HYDRO-B2-02",
            "asset_name": "Hydronic Distribution Manifold B2", "asset_type": "Hydronic Manifold",
            "alert_type": "pressure_deviation", "severity": "MEDIUM",
            "reading": {"parameter": "differential_pressure", "value": 28.5, "unit": "psi",
                        "threshold_min": 10, "threshold_max": 20},
            "location_id": _LOC, "building": "HQ Building A", "floor": "B2",
            "room": "Hydronic Plant Room", "timestamp": "2026-05-06T10:50:00Z",
        },
        # 031 — Process Chiller Unit + refrigerant_charge_low → LLM → HVAC Cooling / Refrig
        {
            "sensor_id": "SNS-031", "asset_id": "SFM-PRCHLR-LAB-01",
            "asset_name": "Process Chiller Lab Building", "asset_type": "Process Chiller Unit",
            "alert_type": "refrigerant_charge_low", "severity": "HIGH",
            "reading": {"parameter": "suction_superheat", "value": 28.0, "unit": "°F",
                        "threshold_max": 15},
            "location_id": _LOC, "building": "Research Lab", "floor": "B1",
            "room": "Lab Plant Room", "timestamp": "2026-05-06T11:00:00Z",
        },
        # 032 — Smart Meter Gateway + data_anomaly → LLM → Energy Management
        {
            "sensor_id": "SNS-032", "asset_id": "SFM-SMGW-PARK-01",
            "asset_name": "Smart Meter Gateway Parking Structure", "asset_type": "Smart Meter Gateway",
            "alert_type": "data_anomaly", "severity": "LOW",
            "reading": {"parameter": "interval_read_error_pct", "value": 35.0, "unit": "%",
                        "threshold_max": 5},
            "location_id": _LOC, "building": "Parking Structure", "floor": "P1",
            "room": "Metering Cabinet", "timestamp": "2026-05-06T11:10:00Z",
        },

        # ══════════════════════════════════════════════════════════════════════
        # TIER 4 — NO MATCH  (asset + alert completely outside service catalog)
        # Expected: No Match → NO_ACTION / log only
        # ══════════════════════════════════════════════════════════════════════

        # 033 — Automated Guided Vehicle: warehouse robot, completely outside FM scope → No Match
        {
            "sensor_id": "SNS-033", "asset_id": "SFM-AGV-WH-03",
            "asset_name": "Automated Guided Vehicle WH-3", "asset_type": "Automated Guided Vehicle",
            "alert_type": "navigation_path_blocked", "severity": "MEDIUM",
            "reading": {"parameter": "obstacle_distance_cm", "value": 8.0, "unit": "cm",
                        "threshold_min": 30},
            "location_id": _LOC, "building": "Warehouse", "floor": "1",
            "room": "Dispatch Bay", "timestamp": "2026-05-06T11:20:00Z",
        },
        # 034 — Coffee machine: completely unrecognized equipment → No Match
        {
            "sensor_id": "SNS-034", "asset_id": "SFM-COFFEE-FL7-01",
            "asset_name": "Commercial Coffee Machine Floor 7", "asset_type": "Coffee Machine",
            "alert_type": "brew_cycle_error", "severity": "LOW",
            "reading": {"parameter": "brew_temp", "value": 175.0, "unit": "°F",
                        "threshold_min": 195, "threshold_max": 205},
            "location_id": _LOC, "building": "HQ Building A", "floor": "7",
            "room": "Kitchen / Break Room", "timestamp": "2026-05-06T11:30:00Z",
        },
        # 035 — Vending machine payment terminal: outside FM scope → No Match
        {
            "sensor_id": "SNS-035", "asset_id": "SFM-VEND-CAFE-02",
            "asset_name": "Vending Machine Cafeteria 2", "asset_type": "Vending Machine",
            "alert_type": "payment_terminal_fault", "severity": "LOW",
            "reading": {"parameter": "transaction_error_count", "value": 14.0, "unit": "errors",
                        "threshold_max": 3},
            "location_id": _LOC, "building": "HQ Building A", "floor": "1",
            "room": "Cafeteria", "timestamp": "2026-05-06T11:40:00Z",
        },
    ]
