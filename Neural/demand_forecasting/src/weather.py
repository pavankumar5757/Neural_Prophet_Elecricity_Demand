from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from config import REGION_TO_CITIES

try:
	 import openmeteo_requests
	 import requests_cache
	 from retry_requests import retry
except Exception:  # optional import guard for docs
	 openmeteo_requests = None


def _client():
	 session = requests_cache.CachedSession(cache_name="openmeteo_cache", expire_after=86400)
	 retry_session = retry(session, retries=3, backoff_factor=0.2)
	 return openmeteo_requests.Client(session=retry_session)


def fetch_regional_weather(start: str, end: Optional[str], regions: Optional[List[str]]):
	 if openmeteo_requests is None:
		 return pd.DataFrame()
	 client = _client()
	 vars_ = [
		 "temperature_2m_max",
		 "temperature_2m_min",
		 "temperature_2m_mean",
		 "precipitation_sum",
		 "wind_speed_10m_max",
		 "relative_humidity_2m_mean",
	 ]
	 rows = []
	 target_regions = regions or list(REGION_TO_CITIES.keys())
	 for region in target_regions:
		 for city, (lat, lon) in REGION_TO_CITIES[region]:
			 url = "https://archive-api.open-meteo.com/v1/era5"
			 params = {
				 "latitude": lat,
				 "longitude": lon,
				 "start_date": start,
				 "end_date": end,
				 "daily": ",".join(vars_),
			 }
			 resp = client.weather_api(url, params=params)
			 d = resp[0].Daily()
			 dates = pd.to_datetime(d.Time(), unit="s")
			 data = {v: d.Variables(vars_.index(v)).ValuesAsNumpy() for v in vars_}
			 df = pd.DataFrame(data, index=dates).reset_index().rename(columns={"index": "ds"})
			 df["region"] = region
			 df["city"] = city
			 rows.append(df)
	 return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def engineer_weather_features(df_weather: pd.DataFrame) -> pd.DataFrame:
	 if df_weather.empty:
		 return df_weather
	 df = df_weather.copy()
	 df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
	 df["HDD"] = (18.0 - df["temperature_2m_mean"]).clip(lower=0.0)
	 df["CDD"] = (df["temperature_2m_mean"] - 24.0).clip(lower=0.0)
	 # Simple apparent temp proxy using mean temp and humidity
	 df["apparent_temp_mean"] = df["temperature_2m_mean"] + 0.05 * (df["relative_humidity_2m_mean"] - 50.0)
	 # Aggregate to region/day
	 agg_cols = [
		 "temp_range",
		 "HDD",
		 "CDD",
		 "apparent_temp_mean",
		 "temperature_2m_mean",
	 ]
	 grouped = df.groupby(["region", "ds"], as_index=False)[agg_cols].mean()
	 return grouped
