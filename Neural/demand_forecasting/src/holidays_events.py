from datetime import date, timedelta
from typing import Set
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


def build_holiday_window_flags(
	 start_year: int = 2016,
	 end_year: int = 2025,
	 pre_days: int = 1,
	 post_days: int = 1,
) -> pd.DataFrame:
	 """
	 Build binary flags for days immediately before and after Indian public holidays.

	 Returns a DataFrame with columns: ds, is_pre_holiday, is_post_holiday
	 Only rows for affected dates are returned; when merging, fill missing with 0.
	 """
	 in_holidays = pyholidays.country_holidays("IN", years=range(start_year, end_year + 1))
	 pre_dates: Set[pd.Timestamp] = set()
	 post_dates: Set[pd.Timestamp] = set()
	 for d in in_holidays:
		 dts = pd.to_datetime(d)
		 if pre_days:
			 for k in range(1, pre_days + 1):
				 pre_dates.add(dts - timedelta(days=k))
		 if post_days:
			 for k in range(1, post_days + 1):
				 post_dates.add(dts + timedelta(days=k))
	 # Aggregate unique dates and emit rows with flags
	 all_dates = sorted(pre_dates.union(post_dates))
	 rows = []
	 for ds in all_dates:
		 rows.append(
			 {
				 "ds": pd.to_datetime(ds),
				 "is_pre_holiday": 1 if ds in pre_dates else 0,
				 "is_post_holiday": 1 if ds in post_dates else 0,
			 }
		 )
	 return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def build_lockdown_flag_df() -> pd.DataFrame:
	 """Create a binary lockdown flag spanning the configured lockdown window."""
	 rows = []
	 current = LOCKDOWN_START
	 while current <= LOCKDOWN_END:
		 rows.append({"ds": pd.to_datetime(current), "is_lockdown": 1})
		 current += timedelta(days=1)
	 return pd.DataFrame(rows)
