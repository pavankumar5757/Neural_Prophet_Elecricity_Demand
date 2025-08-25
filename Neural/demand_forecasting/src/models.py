from typing import Tuple
import pandas as pd
from neuralprophet import NeuralProphet
from src.merge import merge_all
from config import DEFAULT_DEMAND_CSV


def _base_model() -> NeuralProphet:
	 m = NeuralProphet(
		 n_changepoints=20,
		 yearly_seasonality=True,
		 weekly_seasonality=True,
		 daily_seasonality=False,
		 seasonality_mode="additive",
	 )
	 m.add_country_holidays(country_name="IN")
	 return m


def _fit_predict(m: NeuralProphet, df: pd.DataFrame) -> Tuple[NeuralProphet, pd.DataFrame]:
	 model = m.fit(df, freq="D")
	 future = model.make_future_dataframe(df, periods=30)
	 forecast = model.predict(future)
	 return model, forecast


def train_baseline_model(series_id: str) -> Tuple[NeuralProphet, pd.DataFrame]:
	 df = merge_all(DEFAULT_DEMAND_CSV, "2016-01-01", None)
	 df_local = df[df["ID"] == series_id][["ds", "y"]].dropna()
	 m = _base_model()
	 return _fit_predict(m, df_local)


def train_advanced_model(series_id: str) -> Tuple[NeuralProphet, pd.DataFrame]:
	 df = merge_all(DEFAULT_DEMAND_CSV, "2016-01-01", None)
	 df_local = df[df["ID"] == series_id].dropna()
	 m = _base_model()
	 # Autoregression
	 m = m.add_lagged_regressor(names=[], n_lags=0)  # no-op to ensure method availability
	 m.n_lags = 14
	 # Lagged weather regressors
	 for name in ["temp_range", "HDD", "CDD", "apparent_temp_mean", "temperature_2m_mean"]:
		 m.add_lagged_regressor(name, n_lags=7)
	 return _fit_predict(m, df_local[["ds", "y", "temp_range", "HDD", "CDD", "apparent_temp_mean", "temperature_2m_mean"]])


def train_global_model() -> Tuple[NeuralProphet, pd.DataFrame]:
	 df = merge_all(DEFAULT_DEMAND_CSV, "2016-01-01", None)
	 m = _base_model()
	 m = m.add_lagged_regressor(names=[], n_lags=0)
	 m.n_lags = 14
	 for name in ["temp_range", "HDD", "CDD", "apparent_temp_mean", "temperature_2m_mean"]:
		 m.add_lagged_regressor(name, n_lags=7)
	 model, forecast = _fit_predict(m, df[["ds", "y", "ID", "temp_range", "HDD", "CDD", "apparent_temp_mean", "temperature_2m_mean"]])
	 return model, forecast
