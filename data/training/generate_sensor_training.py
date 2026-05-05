"""
generate_sensor_training.py
Generates a 1000+ row labeled training dataset for sensor → service classification.

Output: data/training/sensor_training_data.csv
        data/training/sensor_training_data.json  (first 100 rows as JSON for few-shot)

Columns:
  sensor_id, asset_id, asset_name, asset_type, alert_type,
  reading_value, reading_unit, threshold, building, floor, room,
  service_classification_id, service_classification_name,
  category, priority, match_type, confidence, label
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas not installed — will output JSON only")

random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent
N_ROWS = 1200   # total rows to generate

# ── Catalog stubs (id, name, category, priority) ─────────────────────────────

CLASSIFICATIONS = [
    ("4aa86b28-c506-11ed-afa1-0242ac120002", "HVAC - Cooling System Failure",       "HVAC",             "HIGH"),
    ("5bb97c39-d617-11ed-bfb2-1353bd231113", "HVAC - Heating System Failure",       "HVAC",             "HIGH"),
    ("6cc08d4a-e728-12ee-c0c3-2464ce342224", "HVAC - Thermostat / Controls Issue",  "HVAC",             "MEDIUM"),
    ("7dd19e5b-f839-13ff-d1d4-3575df453335", "HVAC - Ventilation / Air Quality",    "HVAC",             "MEDIUM"),
    ("8ee2af6c-0940-140a-e2e5-4686e056464a", "Electrical - Power Failure",          "Electrical",       "CRITICAL"),
    ("9ff3b07d-1a51-151b-f3f6-5797f167575b", "Electrical - Lighting Issue",         "Electrical",       "LOW"),
    ("a004c18e-2b62-162c-04g7-6808g278686c", "Electrical - Emergency Power",        "Electrical",       "CRITICAL"),
    ("b115d29f-3c73-173d-15h8-7919h389797d", "Plumbing - Water Leak Detection",     "Plumbing",         "CRITICAL"),
    ("c226e3a0-4d84-184e-26i9-8020i490808e", "Plumbing - Domestic Hot Water",       "Plumbing",         "MEDIUM"),
    ("d337f4b1-5e95-195f-37j0-9131j501919f", "Plumbing - Sanitary / Drainage",     "Plumbing",         "HIGH"),
    ("e448g5c2-6f06-106g-48k1-0242k612020g", "Refrigeration - Temperature Variance","Refrigeration",   "HIGH"),
    ("f559h6d3-7g17-117h-59l2-1353l723131h", "BMS/BAS - Control System Fault",     "Building Automation","HIGH"),
    ("g660i7e4-8h28-128i-60m3-2464m834242i", "Fire & Life Safety - Alarm",          "Fire & Safety",    "CRITICAL"),
    ("h771j8f5-9i39-139j-71n4-3575n945353j", "Elevator / Vertical Transport",       "Vertical Transport","CRITICAL"),
    ("i882k9g6-0j40-140k-82o5-4686o056464k", "Mechanical - Vibration / Noise",      "Mechanical",       "MEDIUM"),
    ("j993l0h7-1k51-151l-93p6-5797p167575l", "Mechanical - Pump Failure",           "Mechanical",       "HIGH"),
    ("k004m1i8-2l62-162m-04q7-6808q278686m", "Access Control - Door / Lock Fault",  "Security",         "HIGH"),
    ("l115n2j9-3m73-173n-15r8-7919r389797n", "CCTV / Surveillance - Camera Fault",  "Security",         "MEDIUM"),
    ("m226o3k0-4n84-184o-26s9-8020s490808o", "Energy Management - High Consumption","Energy",           "LOW"),
    ("n337p4l1-5o95-195p-37t0-9131t501919p", "Cooling Tower - Maintenance",         "Mechanical",       "MEDIUM"),
]

# ── Asset → classification mapping (ground truth) ────────────────────────────

ASSET_ALERT_TO_CLASS = {
    # HVAC Cooling
    ("AHU",        "temperature_high"):     0,
    ("FCU",        "temperature_high"):     0,
    ("Chiller",    "cooling_failure"):      0,
    ("CRAC",       "temperature_high"):     0,
    ("AC Unit",    "compressor_fault"):     0,
    # HVAC Heating
    ("AHU",        "temperature_low"):      1,
    ("Boiler",     "heating_failure"):      1,
    ("Heat Pump",  "heat_pump_fault"):      1,
    # Controls
    ("Thermostat", "setpoint_deviation"):   2,
    ("BAS Controller","controls_failure"):  2,
    ("DDC",        "sensor_drift"):         2,
    # Ventilation
    ("AHU",        "co2_high"):             3,
    ("ERU",        "airflow_low"):          3,
    ("Supply Fan", "filter_dirty"):         3,
    # Electrical Power
    ("Electrical Panel","power_failure"):   4,
    ("Switchgear", "breaker_trip"):         4,
    ("MCC",        "voltage_drop"):         4,
    # Lighting
    ("Lighting Panel","lighting_failure"):  5,
    ("Emergency Lighting","lighting_failure"): 5,
    # Emergency Power
    ("Generator",  "generator_fault"):      6,
    ("UPS",        "ups_fault"):            6,
    ("ATS",        "ats_fault"):            6,
    # Water Leak
    ("Water Leak Sensor","water_leak"):     7,
    ("Sump Pump",  "flood_detection"):      7,
    # Hot Water
    ("Water Heater","hot_water_failure"):   8,
    ("Booster Pump","pump_fault"):          8,
    # Drainage
    ("Sewage Pump","drainage_blockage"):    9,
    ("Ejector Pump","pump_fault"):          9,
    # Refrigeration
    ("Freezer",    "freezer_temp_high"):    10,
    ("Walk-in Cooler","fridge_temp_high"):  10,
    ("Refrigeration System","defrost_fault"): 10,
    # BAS
    ("BMS",        "controller_offline"):   11,
    ("SCADA",      "communication_fault"):  11,
    # Fire
    ("Fire Alarm Panel","fire_alarm"):      12,
    ("Smoke Detector","smoke_detection"):   12,
    # Elevator
    ("Elevator",   "elevator_fault"):       13,
    ("Escalator",  "door_fault"):           13,
    # Vibration
    ("Pump",       "vibration_high"):       14,
    ("Compressor", "bearing_fault"):        14,
    ("Motor",      "vibration_high"):       14,
    # Pump Failure
    ("Chilled Water Pump","pump_failure"):  15,
    ("Hot Water Pump","low_flow"):          15,
    # Access Control
    ("Access Control Panel","door_forced"): 16,
    ("Card Reader","reader_offline"):       16,
    # CCTV
    ("CCTV Camera","camera_offline"):       17,
    ("NVR",        "storage_full"):         17,
    # Energy
    ("Energy Meter","energy_spike"):        18,
    ("Sub-meter",  "high_consumption"):     18,
    # Cooling Tower
    ("Cooling Tower","fan_fault"):          19,
    ("Cooling Tower","water_level_low"):    19,
}

ASSET_TYPES = list({k[0] for k in ASSET_ALERT_TO_CLASS})
ALERT_TYPES = list({k[1] for k in ASSET_ALERT_TO_CLASS})

BUILDINGS = [
    "HQ Building A", "HQ Building B", "Tower 1", "Tower 2",
    "Annex North", "Warehouse East", "Data Center", "Parking Structure",
    "Retail Unit R1", "Office Park West",
]
FLOORS   = ["B2", "B1", "G", "1", "2", "3", "4", "5", "Roof"]
ROOMS    = [
    "Mechanical Room", "Electrical Room", "Plant Room", "Server Room",
    "Office Area", "Cafeteria", "Lobby", "Parking", "Rooftop", "Stairwell",
]

MATCH_TYPES  = ["Perfect Match", "Partial Match", "LLM Reasoned", "No Match"]
MATCH_PROBS  = [0.55, 0.25, 0.12, 0.08]   # distribution matches real-world


def _random_reading(alert: str):
    """Generate a plausible sensor reading for the given alert type."""
    reading_map = {
        "temperature_high": (75, 95, 55, 72, "°F"),
        "temperature_low":  (40, 55, 65, 75, "°F"),
        "co2_high":         (900, 1500, 0, 800, "ppm"),
        "vibration_high":   (8, 20, 0, 7, "mm/s"),
        "power_failure":    (0, 10, 100, 250, "V"),
        "voltage_drop":     (180, 210, 210, 240, "V"),
        "water_leak":       (1, 10, 0, 0, "mm"),
        "energy_spike":     (500, 1200, 0, 400, "kW"),
        "freezer_temp_high":(5, 25, -20, 0, "°F"),
        "fridge_temp_high": (42, 55, 32, 40, "°F"),
        "pump_failure":     (0, 20, 50, 120, "GPM"),
        "airflow_low":      (50, 200, 400, 600, "CFM"),
    }
    if alert in reading_map:
        lo, hi, tmin, tmax, unit = reading_map[alert]
        val = round(random.uniform(lo, hi), 1)
        return val, unit, tmin or None, tmax or None
    val = round(random.uniform(10, 100), 1)
    return val, "units", None, None


def _confidence_for(match_type: str) -> float:
    if match_type == "Perfect Match":
        return round(random.uniform(85, 100), 1)
    if match_type == "Partial Match":
        return round(random.uniform(50, 84), 1)
    if match_type == "LLM Reasoned":
        return round(random.uniform(60, 84), 1)
    return 0.0


def generate_rows(n: int) -> list:
    rows = []
    keys = list(ASSET_ALERT_TO_CLASS.keys())

    for i in range(n):
        # 70% of rows use known asset/alert combos for rich ground truth
        if random.random() < 0.70:
            asset_type, alert_type = random.choice(keys)
            class_idx = ASSET_ALERT_TO_CLASS[(asset_type, alert_type)]
            sc = CLASSIFICATIONS[class_idx]
            match_type = random.choices(
                ["Perfect Match", "Partial Match", "LLM Reasoned"],
                weights=[0.65, 0.25, 0.10],
            )[0]
        else:
            # Unknown / edge case combinations → No Match or LLM
            asset_type = random.choice(ASSET_TYPES)
            alert_type = random.choice(ALERT_TYPES)
            # Might still match if combo exists
            class_idx = ASSET_ALERT_TO_CLASS.get((asset_type, alert_type))
            if class_idx is not None:
                sc = CLASSIFICATIONS[class_idx]
                match_type = random.choices(MATCH_TYPES, MATCH_PROBS)[0]
            else:
                sc = random.choice(CLASSIFICATIONS)
                match_type = random.choices(
                    ["LLM Reasoned", "No Match"], weights=[0.4, 0.6]
                )[0]

        val, unit, tmin, tmax = _random_reading(alert_type)
        confidence = _confidence_for(match_type)

        ts = datetime.now(timezone.utc) - timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        rows.append({
            "sensor_id":                    f"SNS-{asset_type.replace(' ','-').upper()}-{i:04d}",
            "asset_id":                     f"SFM-{asset_type.replace(' ','-').upper()}-{random.randint(1,50):03d}",
            "asset_name":                   f"{asset_type} - {random.choice(BUILDINGS)} L{random.choice(FLOORS)}",
            "asset_type":                   asset_type,
            "alert_type":                   alert_type,
            "reading_value":                val,
            "reading_unit":                 unit,
            "threshold_min":                tmin,
            "threshold_max":                tmax,
            "building":                     random.choice(BUILDINGS),
            "floor":                        random.choice(FLOORS),
            "room":                         random.choice(ROOMS),
            "service_classification_id":    sc[0] if match_type != "No Match" else None,
            "service_classification_name":  sc[1] if match_type != "No Match" else "No Match",
            "category":                     sc[2] if match_type != "No Match" else None,
            "priority":                     sc[3] if match_type != "No Match" else None,
            "match_type":                   match_type,
            "confidence":                   confidence,
            "label":                        match_type,
            "timestamp":                    ts.isoformat(),
        })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(N_ROWS)

    # JSON output (first 150 rows as few-shot training examples)
    json_path = OUTPUT_DIR / "sensor_training_data.json"
    with open(json_path, "w") as f:
        json.dump({"total": len(rows), "examples": rows[:150]}, f, indent=2)
    print(f"✅ JSON  → {json_path}  ({min(150, len(rows))} examples)")

    # Full JSONL
    jsonl_path = OUTPUT_DIR / "sensor_training_data.jsonl"
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"✅ JSONL → {jsonl_path}  ({len(rows)} rows)")

    if HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / "sensor_training_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV   → {csv_path}  ({len(df)} rows)")

        # Summary
        print("\n📊 Distribution:")
        print(df["match_type"].value_counts().to_string())
        print("\n📊 By Category:")
        print(df["category"].value_counts().head(10).to_string())
    else:
        from collections import Counter
        ct = Counter(r["match_type"] for r in rows)
        print("\n📊 Distribution:", dict(ct))
