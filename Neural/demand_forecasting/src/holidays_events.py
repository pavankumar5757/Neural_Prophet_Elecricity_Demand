from datetime import date, timedelta
import pandas as pd
import holidays as pyholidays
from config import LOCKDOWN_START, LOCKDOWN_END


def build_holiday_events_df(start_year: int = 2016, end_year: int = 2025) -> pd.DataFrame:
	 in_holidays = pyholidays.country_holidays("IN", years=range(start_year, end_year + 1))
	 rows = []
	 for d, name in in_holidays.items():
		 rows.append({"ds": pd.to_datetime(d), "event": name})
	 return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def build_lockdown_events_df() -> pd.DataFrame:
	 rows = []
	 current = LOCKDOWN_START
	 while current <= LOCKDOWN_END:
		 rows.append({"ds": pd.to_datetime(current), "event": "COVID19_Lockdown"})
		 current += timedelta(days=1)
	 return pd.DataFrame(rows)
