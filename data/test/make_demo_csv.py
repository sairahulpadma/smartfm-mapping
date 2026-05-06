import pandas as pd

rows = [
    ("BLR-01A",         "IFM-001", "Boiler, Steam, Fire Tube(BLR-01A)",   "BLR-01A",    "Pleasanton Campus - Building A",   "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("AHU-02B",         "IFM-002", "Air Handling Unit, Large(AHU-02B)",   "AHU-02B",    "Pleasanton Campus - Building B",   "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("CHILLER-01",      "IFM-003", "Chiller, Centrifugal(CHILLER-01)",    "CHILLER-01", "Chicago HQ - Tower A",             "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("EF-03A",          "IFM-004", "Exhaust Fan, Large(EF-03A)",          "EF-03A",     "Pleasanton Campus - Building A",   "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("PUMP-HW-01",      "IFM-005", "Hot Water Pump(PUMP-HW-01)",          "PUMP-HW-01", "Dallas Office - Main",             "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("RTU-05",          "IFM-006", "Rooftop Package Unit(RTU-05)",        "RTU-05",     "Phoenix Distribution Center",      "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("VAV-412",         "IFM-007", "VAV Box, Single Duct(VAV-412)",       "VAV-412",    "Seattle Campus - West Wing",       "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("CWP-002",         "IFM-008", "Chilled Water Pump(CWP-002)",         "CWP-002",    "NYC Headquarters",                 "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("FCU-301",         "IFM-009", "Fan Coil Unit(FCU-301)",              "FCU-301",    "Atlanta Office Park",              "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    ("COOLING-TWR-1",   "IFM-010", "Cooling Tower(COOLING-TWR-1)",        "COOLING-TWR-1","Columbus Tech Center",           "Perfect - Approach 1", "Perfect - Approach 1", 100.0),
    # Approach 3 perfect (make+serial match)
    ("EF-01B",          "IFM-011", "Fan, Exhaust, Large(EF-BR-1)",        "EF-BR-1",    "Pleasanton Campus - Building B",  "Perfect - Approach 3", "Perfect - Approach 3", 96.7),
    ("RTU 06",          "IFM-013", "Rooftop Package Unit(RTU-PHX-006)",   "RTU-PHX-006","Scottsdale Corporate Campus",      "Perfect - Approach 3", "Perfect - Approach 3", 96.9),
    ("VAV Box 510",     "IFM-014", "VAV Box, Fan Powered(VAV-510)",       "VAV-510",    "Bellevue Tower - Floor 5",         "Perfect - Approach 3", "Perfect - Approach 3", 100.0),
    ("AHU-PENTHOUSE",   "IFM-015", "Air Handling Unit(AHU-PH)",           "AHU-PH",     "Miami Innovation Hub - Penthouse", "Perfect - Approach 3", "Perfect - Approach 3", 100.0),
    # Approach 2 perfect (name + type + location, no make/serial) — LLM verified
    ("COOLING-CTRL-01", "IFM-A01", "Cooling System Controller(COOL-CTRL-01)","COOL-CTRL-01","Savannah Operations Center",   "Perfect - Approach 2 (LLM Verified)", "Approach 2 + LLM", 94.5),
    ("FCU-B2-L01",      "IFM-A02", "Fan Coil Unit B2 Level 1(FCU-B2-L01)","FCU-B2-L01","Minneapolis North Tower",         "Perfect - Approach 2 (LLM Verified)", "Approach 2 + LLM", 96.0),
    # Partial matches — Approach 1 (LLM Verified after partial1 → llm_verify_partial)
    ("PUMP-HTW-03",     "IFM-012", "Hot Water Pump(HWP-03)",              "HWP-03",     "NYC Office Tower - Level B1",      "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 67.4),
    ("HTG-UNIT-3",      "IFM-P01", "Heating Coil Unit 3",                 "HTG-03",     "Denver East Office",               "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 79.6),
    ("SUPPLY-FAN-B2",   "IFM-P02", "Supply Fan Unit SF-B2",               "SF-B2",      "Portland West Building",           "Perfect - Approach 2",                "Approach 2",    100.0),
    ("COND-WATER-PMP-1","IFM-P03", "Condenser Water Pump Loop 1",         "COND-PMP-01","Kansas City Service Depot",        "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 87.8),
    ("XHST-FAN-ROOF-4", "IFM-P04", "Exhaust Fan Roof Unit 04",            "EXH-RF-04",  "Omaha Manufacturing Plant",        "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 78.2),
    ("AIR-COMP-SHOP",   "IFM-P05", "Compressed Air System Shop Floor",    "AIR-COMP-01","Indianapolis Manufacturing",       "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 86.4),
    # Partial + LLM Verified (new)
    ("PUMP-CND-2",      "IFM-A03", "Condenser Water Pump No. 2",          "PUMP-C-002", "Milwaukee Central Utility Plant",  "Partial - Approach 1 (LLM Verified)", "Partial + LLM", 93.0),
    ("BLDG-CTRL-02",    "IFM-A04", "Building Automation Controller 02",   "BAC-02",     "Pittsburgh Campus East Wing",      "Perfect - Approach 3",                "Approach 3",   100.0),
    # LLM Reasoned (original)
    ("HVAC-1",          "IFM-L01", "Air Handling System Unit 1",          "AHS-1",      "Richmond HQ",                      "LLM Reasoned",         "LLM Reasoning",        78.0),
    ("REFRIG-SYS-A",    "IFM-L02", "Refrigeration System Alpha",          "REFRIG-ALPHA","Louisville Cold Storage",         "LLM Reasoned",         "LLM Reasoning",        82.0),
    ("EMER-POWER-1",    "IFM-L03", "Emergency Standby Generator 1",       "ESG-1",      "Columbia Data Center",             "LLM Reasoned",         "LLM Reasoning",        74.0),
    # LLM Reasoned (new — semantic abbreviation gap)
    ("STANDBY-GEN-A",   "IFM-A05", "Emergency Backup Generator Unit Alpha","EBG-ALPHA", "Tucson Data Hub",                  "LLM Reasoned",         "LLM Reasoning",        79.0),
    ("CHILLR-B",        "IFM-A06", "Commercial Water Chiller System B",   "CWC-B",      "Charlotte Cooling Plant",          "LLM Reasoned",         "LLM Reasoning",        81.0),
    # No Match
    ("DECOMMISSIONED-007", None, None, None, None, "No Match", "None", 0.0),
    ("NEW-AHU-999",        None, None, None, None, "No Match", "None", 0.0),
    ("EQ-LEGACY-X1",       None, None, None, None, "No Match", "None", 0.0),
    ("CHILLER-RENTAL-TMP", None, None, None, None, "No Match", "None", 0.0),
    ("UNKNOWN-ASSET-ZZ",   None, None, None, None, "No Match", "None", 0.0),
    ("XFMR-MAIN-480V",     None, None, None, None, "No Match", "None", 0.0),
]

cols = ["sfm_nav_name","matched_asset_id","matched_asset_name","matched_position_name",
        "matched_building","match_type","approach_used","confidence"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("data/test/demo_results.csv", index=False)
print("Saved demo_results.csv")
print(df["match_type"].value_counts())
