from pathlib import Path
import pandas as pd
from src.ingest import load_and_clean_demand_data
from src.holidays_events import (
	build_holiday_events_df,
	build_lockdown_events_df,
	build_holiday_window_flags,
	build_lockdown_flag_df,
)
from src.weather import fetch_regional_weather, engineer_weather_features


def merge_all(demand_csv: Path, start: str, end: str) -> pd.DataFrame:
	wide, long = load_and_clean_demand_data(demand_csv)
	holidays_df = build_holiday_events_df()
	lockdown_df = build_lockdown_events_df()
	events = pd.concat([holidays_df, lockdown_df], ignore_index=True)
	# Additional binary flags for modeling
	holiday_flags = build_holiday_window_flags()
	lockdown_flag = build_lockdown_flag_df()
	weather_raw = fetch_regional_weather(start, end, None)
	weather = engineer_weather_features(weather_raw)
	# Map region column to ID for join
	weather = weather.rename(columns={"region": "ID"})
	df = long.merge(weather, how="left", on=["ID", "ds"])  # lagging will be handled in model
	# Mark events as binary flags (holidays/lockdown specific windows)
	df = df.merge(events.assign(is_event=1), how="left", on="ds")
	df = df.merge(holiday_flags, how="left", on="ds")
	df = df.merge(lockdown_flag, how="left", on="ds")
	for col in ["is_event", "is_pre_holiday", "is_post_holiday", "is_lockdown"]:
		if col in df.columns:
			df[col] = df[col].fillna(0)
	return df
