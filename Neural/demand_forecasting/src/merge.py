from pathlib import Path
import pandas as pd
from src.ingest import load_and_clean_demand_data
from src.holidays_events import build_holiday_events_df, build_lockdown_events_df
from src.weather import fetch_regional_weather, engineer_weather_features


def merge_all(demand_csv: Path, start: str, end: str) -> pd.DataFrame:
	 wide, long = load_and_clean_demand_data(demand_csv)
	 holidays_df = build_holiday_events_df()
	 lockdown_df = build_lockdown_events_df()
	 events = pd.concat([holidays_df, lockdown_df], ignore_index=True)
	 weather_raw = fetch_regional_weather(start, end, None)
	 weather = engineer_weather_features(weather_raw)
	 # Map region column to ID for join
	 weather = weather.rename(columns={"region": "ID"})
	 df = long.merge(weather, how="left", on=["ID", "ds"])  # lagging will be handled in model
	 # Mark events as binary flags (future regressors handled internally by NP)
	 df = df.merge(events.assign(is_event=1), how="left", on="ds")
	 df["is_event"] = df["is_event"].fillna(0)
	 return df
