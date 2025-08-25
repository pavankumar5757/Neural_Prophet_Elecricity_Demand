from pathlib import Path
from datetime import date
from typing import Dict, Tuple, List

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
for d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR, FIG_DIR):
	 d.mkdir(parents=True, exist_ok=True)

# Input files
DEFAULT_DEMAND_CSV = DATA_DIR / "daily_generation_and_demand.csv"

# Regions and IDs
REGION_IDS: List[str] = [
	 "NR_Demand",
	 "WR_Demand",
	 "SR_Demand",
	 "ER_Demand",
	 "NER_Demand",
	 "Country_Demand",
]

# Weather proxy mapping: region -> [(city, (lat, lon)), ...]
REGION_TO_CITIES: Dict[str, List[Tuple[str, Tuple[float, float]]]] = {
	 "NR_Demand": [("Delhi", (28.61, 77.23)), ("Jaipur", (26.91, 75.78))],
	 "WR_Demand": [("Mumbai", (19.07, 72.87)), ("Ahmedabad", (23.02, 72.57))],
	 "SR_Demand": [("Chennai", (13.08, 80.27)), ("Bengaluru", (12.97, 77.59))],
	 "ER_Demand": [("Kolkata", (22.57, 88.36)), ("Patna", (25.59, 85.13))],
	 "NER_Demand": [("Guwahati", (26.14, 91.73)), ("Shillong", (25.57, 91.89))],
	 "Country_Demand": [("Delhi", (28.61, 77.23))],
}

# Lockdown event window (inclusive)
LOCKDOWN_START = date(2020, 3, 25)
LOCKDOWN_END = date(2020, 5, 31)
