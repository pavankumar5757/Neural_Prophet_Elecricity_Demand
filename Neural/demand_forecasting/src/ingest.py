from pathlib import Path
from typing import Tuple
import pandas as pd

REQUIRED_COLUMNS = [
	 "Date",
	 "NR_Demand",
	 "WR_Demand",
	 "SR_Demand",
	 "ER_Demand",
	 "NER_Demand",
	 "Country_Demand",
]


def _read_csv_resilient(file_path: Path) -> pd.DataFrame:
	 # Try to auto-detect header by finding the row that contains "Date"
	 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		 lines = f.readlines()
	 header_idx = None
	 for i, line in enumerate(lines[:20]):
		 if "Date" in line and "," in line:
			 header_idx = i
			 break
	 df = pd.read_csv(file_path, header=header_idx)
	 # Drop footer summary rows: keep rows where Date parses
	 df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
	 df = df.dropna(subset=["Date"]).copy()
	 return df


def load_and_clean_demand_data(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
	 df = _read_csv_resilient(file_path)
	 # Keep only required columns if present
	 present = [c for c in REQUIRED_COLUMNS if c in df.columns]
	 df = df[present].copy()
	 # Coerce numeric on demand columns
	 for c in present:
		 if c != "Date":
			 df[c] = pd.to_numeric(df[c], errors="coerce")
	 # Wide -> Long for global modeling
	 long = df.melt(id_vars=["Date"], var_name="ID", value_name="y").rename(columns={"Date": "ds"})
	 long = long.dropna(subset=["y"]).sort_values(["ID", "ds"]).reset_index(drop=True)
	 return df.sort_values("Date").reset_index(drop=True), long
