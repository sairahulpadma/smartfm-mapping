"""
generate_test_data.py
Generates two demo Excel files for the hackathon:

  test_sfm_demo.xlsx  – 20 SFM assets (source system)
  test_ifm_demo.xlsx  – 25 IFM assets (target system)

Ground truth:
  - 10 perfect matches (high confidence)
  -  5 partial matches (medium confidence)
  -  5 no-matches     (SFM records with no IFM counterpart)

Run: python3 data/test/generate_test_data.py
"""

import os
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── SFM Records (20 total) ────────────────────────────────────────────────────
SFM_DATA = [
    # ── Perfect matches ──────────────────────────────────────────────────────
    {
        "nav_name": "BLR-01A", "equip_type": "Boiler",
        "equip_make": "Raypak", "equip_model": "H7-1505A",
        "equip_serial": "SN100001",
        "country": "United States", "state": "CA", "city": "Pleasanton",
        "site_name": "Pleasanton Campus - Building A",
    },
    {
        "nav_name": "AHU-02B", "equip_type": "Air Handling Unit",
        "equip_make": "Trane", "equip_model": "CLCH-D24",
        "equip_serial": "SN100002",
        "country": "United States", "state": "CA", "city": "Pleasanton",
        "site_name": "Pleasanton Campus - Building B",
    },
    {
        "nav_name": "CHILLER-01", "equip_type": "Chiller",
        "equip_make": "Carrier", "equip_model": "30XA-400",
        "equip_serial": "SN100003",
        "country": "United States", "state": "IL", "city": "Chicago",
        "site_name": "Chicago HQ - Tower A",
    },
    {
        "nav_name": "EF-03A", "equip_type": "Exhaust Fan",
        "equip_make": "Centrimaster", "equip_model": "CFM-1200",
        "equip_serial": "SN100004",
        "country": "United States", "state": "CA", "city": "Pleasanton",
        "site_name": "Pleasanton Campus - Building A",
    },
    {
        "nav_name": "PUMP-HW-01", "equip_type": "Pump",
        "equip_make": "Grundfos", "equip_model": "CM5-6",
        "equip_serial": "SN100005",
        "country": "United States", "state": "TX", "city": "Dallas",
        "site_name": "Dallas Office - Main",
    },
    {
        "nav_name": "RTU-05", "equip_type": "Rooftop Unit",
        "equip_make": "Lennox", "equip_model": "LGH180",
        "equip_serial": "SN100006",
        "country": "United States", "state": "AZ", "city": "Phoenix",
        "site_name": "Phoenix Distribution Center",
    },
    {
        "nav_name": "VAV-412", "equip_type": "VAV",
        "equip_make": "Johnson Controls", "equip_model": "VAV-SD",
        "equip_serial": "SN100007",
        "country": "United States", "state": "WA", "city": "Seattle",
        "site_name": "Seattle Campus - West Wing",
    },
    {
        "nav_name": "CWP-002", "equip_type": "Chilled Water Pump",
        "equip_make": "Bell & Gossett", "equip_model": "e-1510",
        "equip_serial": "SN100008",
        "country": "United States", "state": "NY", "city": "New York",
        "site_name": "NYC Headquarters",
    },
    {
        "nav_name": "FCU-301", "equip_type": "Fan Coil Unit",
        "equip_make": "Daikin", "equip_model": "FWF04",
        "equip_serial": "SN100009",
        "country": "United States", "state": "GA", "city": "Atlanta",
        "site_name": "Atlanta Office Park",
    },
    {
        "nav_name": "COOLING-TWR-1", "equip_type": "Cooling Tower",
        "equip_make": "Baltimore Aircoil", "equip_model": "VTL-322",
        "equip_serial": "SN100010",
        "country": "United States", "state": "OH", "city": "Columbus",
        "site_name": "Columbus Tech Center",
    },
    # ── Partial matches ───────────────────────────────────────────────────────
    {
        "nav_name": "EF-01B", "equip_type": "Exhaust Fan",
        "equip_make": "UNK", "equip_model": "UNK",
        "equip_serial": "UNK-SN11",
        "country": "United States", "state": "CA", "city": "Pleasanton",
        "site_name": "Pleasanton Campus Bldg B",   # slight mismatch
    },
    {
        "nav_name": "PUMP-HTW-03", "equip_type": "Pump",
        "equip_make": "", "equip_model": "",
        "equip_serial": "",
        "country": "United States", "state": "NY", "city": "New York",
        "site_name": "NYC Office Tower",
    },
    {
        "nav_name": "RTU 06", "equip_type": "Rooftop Unit",
        "equip_make": "Lennox", "equip_model": "LGH240",
        "equip_serial": "SN200001",
        "country": "United States", "state": "AZ", "city": "Scottsdale",
        "site_name": "Scottsdale Corp Campus",
    },
    {
        "nav_name": "VAV Box 510", "equip_type": "VAV",
        "equip_make": "Siemens", "equip_model": "VAV-FD",
        "equip_serial": "",
        "country": "United States", "state": "WA", "city": "Bellevue",
        "site_name": "Bellevue Tower",
    },
    {
        "nav_name": "AHU-PENTHOUSE", "equip_type": "Air Handling Unit",
        "equip_make": "York", "equip_model": "YCAL0048",
        "equip_serial": "SN200002",
        "country": "United States", "state": "FL", "city": "Miami",
        "site_name": "Miami Innovation Hub - Penthouse",
    },
    # ── True Partial matches (designed to score 40–69%) ──────────────────────
    # Name similarity ~55%, location matches, building partially matches
    {
        "nav_name": "HTG-UNIT-3",          # vs IFM "Heating Coil 3" → ~55% name match
        "equip_type": "Heating Unit",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "CO", "city": "Denver",
        "site_name": "Denver East Campus",  # vs "Denver East Office" → ~75% building
    },
    {
        "nav_name": "SUPPLY-FAN-B2",       # vs IFM "SF-B2" → ~52% name match
        "equip_type": "Supply Fan",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "OR", "city": "Portland",
        "site_name": "Portland West Site",  # vs "Portland West Building" → ~72% building
    },
    {
        "nav_name": "COND-WATER-PMP-1",    # vs IFM "CWP-Loop-1" → ~50% name match
        "equip_type": "Pump",
        "equip_make": "Taco", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "MO", "city": "Kansas City",
        "site_name": "KC Service Depot",    # vs "Kansas City Service Depot" → ~70% building
    },
    {
        "nav_name": "XHST-FAN-ROOF-4",     # vs IFM "EXH-RF-04" → ~52% name match
        "equip_type": "Exhaust Fan",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "NE", "city": "Omaha",
        "site_name": "Omaha Plant Bldg 2",  # vs "Omaha Manufacturing Plant" → ~62% building
    },
    {
        "nav_name": "AIR-COMP-SHOP",       # vs IFM "Compressor Shop Floor" → ~55% name
        "equip_type": "Air Compressor",
        "equip_make": "Ingersoll Rand", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "IN", "city": "Indianapolis",
        "site_name": "Indy Manufacturing",  # vs "Indianapolis Manufacturing Facility" → ~68%
    },
    # ── LLM Reasoned (name too different for fuzzy, NO make/model/serial to prevent Approach 3) ─
    # These require LLM to recognise e.g. "HVAC-1" == "Air Handling System Unit 1"
    {
        "nav_name": "HVAC-1",              # LLM: matches "Air Handling System Unit 1"
        "equip_type": "Air Handling Unit",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "VA", "city": "Richmond",
        "site_name": "Richmond HQ",
    },
    {
        "nav_name": "REFRIG-SYS-A",        # LLM: matches "Refrigeration System Alpha"
        "equip_type": "Refrigeration",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "KY", "city": "Louisville",
        "site_name": "Louisville Cold Storage",
    },
    {
        "nav_name": "EMER-POWER-1",        # LLM: matches "Emergency Standby Generator 1"
        "equip_type": "Generator",
        "equip_make": "", "equip_model": "", "equip_serial": "",
        "country": "United States", "state": "SC", "city": "Columbia",
        "site_name": "Columbia Data Center",
    },
    # ── No match (intentionally unmatched) ───────────────────────────────────
    {
        "nav_name": "DECOMMISSIONED-007", "equip_type": "Generator",
        "equip_make": "Caterpillar", "equip_model": "3516C",
        "equip_serial": "CAT2005007",
        "country": "United States", "state": "CA", "city": "San Francisco",
        "site_name": "SF HQ Basement",
    },
    {
        "nav_name": "NEW-AHU-999", "equip_type": "Air Handling Unit",
        "equip_make": "Carrier", "equip_model": "39HQ",
        "equip_serial": "CAR2026999",
        "country": "United States", "state": "FL", "city": "Tampa",
        "site_name": "Tampa New Build",
    },
    {
        "nav_name": "EQ-LEGACY-X1", "equip_type": "",
        "equip_make": "", "equip_model": "",
        "equip_serial": "",
        "country": "", "state": "", "city": "",
        "site_name": "",
    },
    {
        "nav_name": "CHILLER-RENTAL-TMP", "equip_type": "Chiller",
        "equip_make": "Aggreko", "equip_model": "TEMP-UNIT",
        "equip_serial": "AGG2025TMP",
        "country": "United States", "state": "TX", "city": "Houston",
        "site_name": "Houston Temporary Site",
    },
    {
        "nav_name": "UNKNOWN-ASSET-ZZ", "equip_type": "Other",
        "equip_make": "Unknown", "equip_model": "Unknown",
        "equip_serial": "ZZ9999",
        "country": "United States", "state": "NV", "city": "Las Vegas",
        "site_name": "Las Vegas Warehouse",
    },
]

# ── IFM Records (25 total — includes 10 perfect + 5 partial + 10 extras) ─────
IFM_DATA = [
    # ── Perfect match targets ─────────────────────────────────────────────────
    {
        "asset_id": "IFM-001", "asset_alternate_id": "ALT-001",
        "asset_name": "Boiler, Steam, Fire Tube(BLR-01A)",
        "asset_status": "ACTIVE", "manufacturer": "Raypak",
        "serial_number": "SN100001", "model": "H7-1505A",
        "equip_part_description": "Boiler, Steam, Fire Tube",
        "position_name": "BLR-01A",
        "position_type_description": "Boiler",
        "region_name": "US, CA, Pleasanton",
        "building_name": "Pleasanton Campus - Building A",
        "floor_name": "Roof", "room_name": "Mechanical",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-002", "asset_alternate_id": "ALT-002",
        "asset_name": "Air Handling Unit, Large(AHU-02B)",
        "asset_status": "ACTIVE", "manufacturer": "Trane",
        "serial_number": "SN100002", "model": "CLCH-D24",
        "equip_part_description": "Air Handling Unit, Large",
        "position_name": "AHU-02B",
        "position_type_description": "Air Handling Unit",
        "region_name": "US, CA, Pleasanton",
        "building_name": "Pleasanton Campus - Building B",
        "floor_name": "Level 1", "room_name": "Mechanical Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-003", "asset_alternate_id": "ALT-003",
        "asset_name": "Chiller, Centrifugal, Water Cooled(CHILLER-01)",
        "asset_status": "ACTIVE", "manufacturer": "Carrier",
        "serial_number": "SN100003", "model": "30XA-400",
        "equip_part_description": "Chiller, Centrifugal",
        "position_name": "CHILLER-01",
        "position_type_description": "Chiller",
        "region_name": "US, IL, Chicago",
        "building_name": "Chicago HQ - Tower A",
        "floor_name": "Basement", "room_name": "Central Plant",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-004", "asset_alternate_id": "ALT-004",
        "asset_name": "Exhaust Fan, Large(EF-03A)",
        "asset_status": "ACTIVE", "manufacturer": "Centrimaster",
        "serial_number": "SN100004", "model": "CFM-1200",
        "equip_part_description": "Exhaust Fan, Large",
        "position_name": "EF-03A",
        "position_type_description": "Exhaust Fan",
        "region_name": "US, CA, Pleasanton",
        "building_name": "Pleasanton Campus - Building A",
        "floor_name": "Roof", "room_name": "Roof Level",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-005", "asset_alternate_id": "ALT-005",
        "asset_name": "Hot Water Pump(PUMP-HW-01)",
        "asset_status": "ACTIVE", "manufacturer": "Grundfos",
        "serial_number": "SN100005", "model": "CM5-6",
        "equip_part_description": "Pump, Hot Water",
        "position_name": "PUMP-HW-01",
        "position_type_description": "Pump",
        "region_name": "US, TX, Dallas",
        "building_name": "Dallas Office - Main",
        "floor_name": "Level B1", "room_name": "Pump Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-006", "asset_alternate_id": "ALT-006",
        "asset_name": "Rooftop Package Unit(RTU-05)",
        "asset_status": "ACTIVE", "manufacturer": "Lennox",
        "serial_number": "SN100006", "model": "LGH180",
        "equip_part_description": "Rooftop Unit",
        "position_name": "RTU-05",
        "position_type_description": "Rooftop Unit",
        "region_name": "US, AZ, Phoenix",
        "building_name": "Phoenix Distribution Center",
        "floor_name": "Roof", "room_name": "Roof",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-007", "asset_alternate_id": "ALT-007",
        "asset_name": "VAV Box, Single Duct(VAV-412)",
        "asset_status": "ACTIVE", "manufacturer": "Johnson Controls",
        "serial_number": "SN100007", "model": "VAV-SD",
        "equip_part_description": "VAV Box, Single Duct",
        "position_name": "VAV-412",
        "position_type_description": "VAV",
        "region_name": "US, WA, Seattle",
        "building_name": "Seattle Campus - West Wing",
        "floor_name": "Floor 4", "room_name": "Open Office",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-008", "asset_alternate_id": "ALT-008",
        "asset_name": "Chilled Water Pump(CWP-002)",
        "asset_status": "ACTIVE", "manufacturer": "Bell & Gossett",
        "serial_number": "SN100008", "model": "e-1510",
        "equip_part_description": "Chilled Water Pump",
        "position_name": "CWP-002",
        "position_type_description": "Chilled Water Pump",
        "region_name": "US, NY, New York",
        "building_name": "NYC Headquarters",
        "floor_name": "Basement", "room_name": "Central Plant",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-009", "asset_alternate_id": "ALT-009",
        "asset_name": "Fan Coil Unit(FCU-301)",
        "asset_status": "ACTIVE", "manufacturer": "Daikin",
        "serial_number": "SN100009", "model": "FWF04",
        "equip_part_description": "Fan Coil Unit",
        "position_name": "FCU-301",
        "position_type_description": "Fan Coil Unit",
        "region_name": "US, GA, Atlanta",
        "building_name": "Atlanta Office Park",
        "floor_name": "Floor 3", "room_name": "Conference Zone",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-010", "asset_alternate_id": "ALT-010",
        "asset_name": "Cooling Tower(COOLING-TWR-1)",
        "asset_status": "ACTIVE", "manufacturer": "Baltimore Aircoil",
        "serial_number": "SN100010", "model": "VTL-322",
        "equip_part_description": "Cooling Tower",
        "position_name": "COOLING-TWR-1",
        "position_type_description": "Cooling Tower",
        "region_name": "US, OH, Columbus",
        "building_name": "Columbus Tech Center",
        "floor_name": "Roof", "room_name": "Cooling Plant",
        "customer_name": "Clorox",
    },
    # ── Partial match targets ─────────────────────────────────────────────────
    {
        "asset_id": "IFM-011", "asset_alternate_id": "ALT-011",
        "asset_name": "Fan, Exhaust, Large(EF-BR-1)",
        "asset_status": "DRAFT", "manufacturer": "UNK",
        "serial_number": "UNK-SN11", "model": "UNK",
        "equip_part_description": "Exhaust Fan",
        "position_name": "EF-BR-1",
        "position_type_description": "Exhaust Fan",
        "region_name": "US, CA, Pleasanton",
        "building_name": "Pleasanton Campus - Building B",
        "floor_name": "Roof", "room_name": "Roof",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-012", "asset_alternate_id": "ALT-012",
        "asset_name": "Hot Water Pump(HWP-03)",
        "asset_status": "ACTIVE", "manufacturer": "",
        "serial_number": "", "model": "",
        "equip_part_description": "Hot Water Pump",
        "position_name": "HWP-03",
        "position_type_description": "Pump",
        "region_name": "US, NY, New York",
        "building_name": "NYC Office Tower - Level B1",
        "floor_name": "Level B1", "room_name": "Pump Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-013", "asset_alternate_id": "ALT-013",
        "asset_name": "Rooftop Package Unit(RTU-PHX-006)",
        "asset_status": "ACTIVE", "manufacturer": "Lennox",
        "serial_number": "SN200001", "model": "LGH240",
        "equip_part_description": "Rooftop Unit",
        "position_name": "RTU-PHX-006",
        "position_type_description": "Rooftop Unit",
        "region_name": "US, AZ, Scottsdale",
        "building_name": "Scottsdale Corporate Campus",
        "floor_name": "Roof", "room_name": "Roof",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-014", "asset_alternate_id": "ALT-014",
        "asset_name": "VAV Box, Fan Powered(VAV-510)",
        "asset_status": "ACTIVE", "manufacturer": "Siemens",
        "serial_number": "", "model": "VAV-FD",
        "equip_part_description": "VAV Box, Fan Powered",
        "position_name": "VAV-510",
        "position_type_description": "VAV",
        "region_name": "US, WA, Bellevue",
        "building_name": "Bellevue Tower - Floor 5",
        "floor_name": "Floor 5", "room_name": "East Zone",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-015", "asset_alternate_id": "ALT-015",
        "asset_name": "Air Handling Unit(AHU-PH)",
        "asset_status": "ACTIVE", "manufacturer": "York",
        "serial_number": "SN200002", "model": "YCAL0048",
        "equip_part_description": "Air Handling Unit",
        "position_name": "AHU-PH",
        "position_type_description": "Air Handling Unit",
        "region_name": "US, FL, Miami",
        "building_name": "Miami Innovation Hub - Penthouse Level",
        "floor_name": "Penthouse", "room_name": "Mechanical",
        "customer_name": "Clorox",
    },
    # ── Partial match IFM targets (deliberately different names) ─────────────
    {
        "asset_id": "IFM-P01", "asset_alternate_id": "ALTP-01",
        "asset_name": "Heating Coil Unit 3",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Heating Coil",
        "position_name": "HC-3",
        "position_type_description": "Heating Unit",
        "region_name": "US, CO, Denver",
        "building_name": "Denver East Office",
        "floor_name": "Floor 2", "room_name": "Mechanical", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-P02", "asset_alternate_id": "ALTP-02",
        "asset_name": "Supply Fan Unit SF-B2",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Supply Fan",
        "position_name": "SF-B2",
        "position_type_description": "Supply Fan",
        "region_name": "US, OR, Portland",
        "building_name": "Portland West Building",
        "floor_name": "Roof", "room_name": "Rooftop", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-P03", "asset_alternate_id": "ALTP-03",
        "asset_name": "Condenser Water Pump Loop 1",
        "asset_status": "ACTIVE", "manufacturer": "Taco", "serial_number": "", "model": "",
        "equip_part_description": "Condenser Water Pump",
        "position_name": "CWP-Loop-1",
        "position_type_description": "Pump",
        "region_name": "US, MO, Kansas City",
        "building_name": "Kansas City Service Depot",
        "floor_name": "Level B1", "room_name": "Pump Room", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-P04", "asset_alternate_id": "ALTP-04",
        "asset_name": "Exhaust Fan Roof Unit 04",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Exhaust Fan",
        "position_name": "EXH-RF-04",
        "position_type_description": "Exhaust Fan",
        "region_name": "US, NE, Omaha",
        "building_name": "Omaha Manufacturing Plant",
        "floor_name": "Roof", "room_name": "Roof", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-P05", "asset_alternate_id": "ALTP-05",
        "asset_name": "Compressed Air System Shop Floor",
        "asset_status": "ACTIVE", "manufacturer": "Ingersoll Rand",
        "serial_number": "", "model": "",
        "equip_part_description": "Air Compressor",
        "position_name": "AIR-COMP-01",
        "position_type_description": "Air Compressor",
        "region_name": "US, IN, Indianapolis",
        "building_name": "Indianapolis Manufacturing Facility",
        "floor_name": "Floor 1", "room_name": "Compressor Room", "customer_name": "Clorox",
    },
    # ── LLM Reasoned IFM targets (name semantically same but text very different) ─
    # IFM-L01/L02/L03: blank make/model/serial so Approach 3 cannot fire.
    # Name is semantically same as SFM but textually unrecognisable by fuzzy scorers.
    # Only the LLM can bridge "HVAC-1" → "Air Handling System Unit 1", etc.
    {
        "asset_id": "IFM-L01", "asset_alternate_id": "ALTL-01",
        "asset_name": "Air Handling System Unit 1",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Air Handling System",
        "position_name": "AHS-1",
        "position_type_description": "Air Handling Unit",
        "region_name": "US, VA, Richmond",
        "building_name": "Richmond HQ",
        "floor_name": "Level 3", "room_name": "Mechanical", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-L02", "asset_alternate_id": "ALTL-02",
        "asset_name": "Refrigeration System Alpha",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Refrigeration System",
        "position_name": "REFRIG-ALPHA",
        "position_type_description": "Refrigeration",
        "region_name": "US, KY, Louisville",
        "building_name": "Louisville Cold Storage",
        "floor_name": "Level B1", "room_name": "Cold Room", "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-L03", "asset_alternate_id": "ALTL-03",
        "asset_name": "Emergency Standby Generator 1",
        "asset_status": "ACTIVE", "manufacturer": "", "serial_number": "", "model": "",
        "equip_part_description": "Emergency Generator",
        "position_name": "ESG-1",
        "position_type_description": "Generator",
        "region_name": "US, SC, Columbia",
        "building_name": "Columbia Data Center",
        "floor_name": "Ground", "room_name": "Generator Pad", "customer_name": "Clorox",
    },
    # ── Extra IFM-only records ────────────────────────────────────────────────
    {
        "asset_id": "IFM-016", "asset_alternate_id": "ALT-016",
        "asset_name": "Generator, Emergency(GEN-001)",
        "asset_status": "ACTIVE", "manufacturer": "Cummins",
        "serial_number": "CUM2020001", "model": "C750D5",
        "equip_part_description": "Emergency Generator",
        "position_name": "GEN-001",
        "position_type_description": "Generator",
        "region_name": "US, CA, Oakland",
        "building_name": "Oakland Data Center",
        "floor_name": "Ground", "room_name": "Generator Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-017", "asset_alternate_id": "ALT-017",
        "asset_name": "Fire Suppression System(FSS-A1)",
        "asset_status": "ACTIVE", "manufacturer": "Tyco",
        "serial_number": "TYC2019A1", "model": "Ansul-R102",
        "equip_part_description": "Fire Suppression",
        "position_name": "FSS-A1",
        "position_type_description": "Fire Suppression",
        "region_name": "US, CA, San Jose",
        "building_name": "San Jose Research Lab",
        "floor_name": "All Floors", "room_name": "Server Rooms",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-018", "asset_alternate_id": "ALT-018",
        "asset_name": "Elevator, Passenger(ELEV-01)",
        "asset_status": "ACTIVE", "manufacturer": "Otis",
        "serial_number": "OTS2015001", "model": "Gen2-MRL",
        "equip_part_description": "Passenger Elevator",
        "position_name": "ELEV-01",
        "position_type_description": "Elevator",
        "region_name": "US, CO, Denver",
        "building_name": "Denver Office Tower",
        "floor_name": "All", "room_name": "Elevator Shaft",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-019", "asset_alternate_id": "ALT-019",
        "asset_name": "UPS System(UPS-DC-01)",
        "asset_status": "ACTIVE", "manufacturer": "Eaton",
        "serial_number": "EAT2021DC1", "model": "9PX6KiPM",
        "equip_part_description": "Uninterruptible Power Supply",
        "position_name": "UPS-DC-01",
        "position_type_description": "UPS",
        "region_name": "US, CA, Oakland",
        "building_name": "Oakland Data Center",
        "floor_name": "Level 1", "room_name": "Server Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-020", "asset_alternate_id": "ALT-020",
        "asset_name": "Air Compressor(AC-01)",
        "asset_status": "INACTIVE", "manufacturer": "Atlas Copco",
        "serial_number": "AC2016001", "model": "GA30",
        "equip_part_description": "Air Compressor",
        "position_name": "AC-01",
        "position_type_description": "Air Compressor",
        "region_name": "US, WI, Milwaukee",
        "building_name": "Milwaukee Plant",
        "floor_name": "Floor 1", "room_name": "Compressor Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-021", "asset_alternate_id": "ALT-021",
        "asset_name": "Cooling Tower(CT-02)",
        "asset_status": "ACTIVE", "manufacturer": "Evapco",
        "serial_number": "EVP2018002", "model": "AT-300",
        "equip_part_description": "Cooling Tower",
        "position_name": "CT-02",
        "position_type_description": "Cooling Tower",
        "region_name": "US, NC, Charlotte",
        "building_name": "Charlotte Distribution Hub",
        "floor_name": "Roof", "room_name": "Rooftop",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-022", "asset_alternate_id": "ALT-022",
        "asset_name": "Boiler, Hot Water(BOILER-HW-1)",
        "asset_status": "ACTIVE", "manufacturer": "Cleaver-Brooks",
        "serial_number": "CB2020001", "model": "FLX-100",
        "equip_part_description": "Hot Water Boiler",
        "position_name": "BOILER-HW-1",
        "position_type_description": "Boiler",
        "region_name": "US, MN, Minneapolis",
        "building_name": "Minneapolis HQ",
        "floor_name": "Basement", "room_name": "Boiler Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-023", "asset_alternate_id": "ALT-023",
        "asset_name": "Transformer, Dry Type(XFMR-01)",
        "asset_status": "ACTIVE", "manufacturer": "ABB",
        "serial_number": "ABB2017001", "model": "RESIBLOC",
        "equip_part_description": "Dry Type Transformer",
        "position_name": "XFMR-01",
        "position_type_description": "Electrical Transformer",
        "region_name": "US, OR, Portland",
        "building_name": "Portland Tech Hub",
        "floor_name": "Ground", "room_name": "Electrical Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-024", "asset_alternate_id": "ALT-024",
        "asset_name": "Split System AC(SPLIT-B3-07)",
        "asset_status": "ACTIVE", "manufacturer": "Mitsubishi",
        "serial_number": "MIT2022B07", "model": "PUY-A24",
        "equip_part_description": "Split System Air Conditioner",
        "position_name": "SPLIT-B3-07",
        "position_type_description": "Split System AC",
        "region_name": "US, NM, Albuquerque",
        "building_name": "Albuquerque Service Center",
        "floor_name": "Floor 3", "room_name": "Server Room",
        "customer_name": "Clorox",
    },
    {
        "asset_id": "IFM-025", "asset_alternate_id": "ALT-025",
        "asset_name": "Heat Exchanger(HX-LOOP-1)",
        "asset_status": "ACTIVE", "manufacturer": "Alfa Laval",
        "serial_number": "AL2019001", "model": "M10-BFM",
        "equip_part_description": "Plate Heat Exchanger",
        "position_name": "HX-LOOP-1",
        "position_type_description": "Heat Exchanger",
        "region_name": "US, MA, Boston",
        "building_name": "Boston Innovation Center",
        "floor_name": "Level B2", "room_name": "Mechanical Room",
        "customer_name": "Clorox",
    },
]


def generate():
    sfm_df = pd.DataFrame(SFM_DATA)
    ifm_df = pd.DataFrame(IFM_DATA)

    sfm_path = os.path.join(OUT_DIR, "test_sfm_demo.xlsx")
    ifm_path = os.path.join(OUT_DIR, "test_ifm_demo.xlsx")

    sfm_df.to_excel(sfm_path, index=False)
    ifm_df.to_excel(ifm_path, index=False)

    print(f"✅ Generated {sfm_path}  ({len(sfm_df)} SFM records)")
    print(f"✅ Generated {ifm_path}  ({len(ifm_df)} IFM records)")
    print()
    print("Expected results:")
    print("  🟢 10 Perfect matches   (BLR-01A → COOLING-TWR-1)")
    print("  🟢  5 Perfect-Approach3 (EF-01B, RTU-06, VAV Box 510, AHU-PENTHOUSE + 1)")
    print("  🟡  5 Partial matches   (40–69%)  → HTG-UNIT-3 through AIR-COMP-SHOP")
    print("  🔵  3 LLM Reasoned      (HVAC-1, REFRIG-SYS-A, EMER-POWER-1) — needs LLM API key")
    print("  🔴  5 No matches        (DECOMMISSIONED-007 through UNKNOWN-ASSET-ZZ)")
    return sfm_path, ifm_path


if __name__ == "__main__":
    generate()
