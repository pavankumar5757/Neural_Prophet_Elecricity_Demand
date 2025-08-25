from typing import Dict, Any
import itertools
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from src.merge import merge_all
from config import DEFAULT_DEMAND_CSV


def grid_search_tuning(series_id: str, max_trials: int = 20) -> Dict[str, Any]:
	 df = merge_all(DEFAULT_DEMAND_CSV, "2016-01-01", None)
	 df_local = df[df["ID"] == series_id][["ds", "y"]].dropna().reset_index(drop=True)
	 param_grid = {
		 "n_lags": [7, 14, 21],
		 "n_changepoints": [10, 20, 40],
		 "learning_rate": [0.001, 0.01, 0.1],
	 }
	 combos = list(itertools.product(*param_grid.values()))[:max_trials]
	 best = {"RMSE": float("inf")}
	 for n_lags, n_changepoints, lr in combos:
		 m = NeuralProphet(
			 n_changepoints=n_changepoints,
			 yearly_seasonality=True,
			 weekly_seasonality=True,
			 learning_rate=lr,
		 )
		 if n_lags:
			 m.n_lags = n_lags
		 # simple holdout
		 split = int(len(df_local) * 0.8)
		 train = df_local.iloc[:split]
		 valid = df_local.iloc[split:]
		 m.fit(train, freq="D", progress_bar=False)
		 fcst = m.predict(m.make_future_dataframe(train, periods=len(valid)))
		 yhat = fcst.tail(len(valid))["yhat1"].to_numpy()
		 rmse = float(np.sqrt(np.mean((valid["y"].to_numpy() - yhat) ** 2)))
		 if rmse < best["RMSE"]:
			 best = {"RMSE": rmse, "n_lags": n_lags, "n_changepoints": n_changepoints, "learning_rate": lr}
	 return best
