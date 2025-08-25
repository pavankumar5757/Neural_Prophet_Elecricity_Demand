from typing import Dict
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from src.merge import merge_all
from config import DEFAULT_DEMAND_CSV


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	 mae = float(np.mean(np.abs(y_true - y_pred)))
	 rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
	 return {"MAE": mae, "RMSE": rmse}


def rolling_origin_cv(series_id: str, horizon: int = 30, step: int = 30) -> Dict[str, float]:
	 df = merge_all(DEFAULT_DEMAND_CSV, "2016-01-01", None)
	 df_local = df[df["ID"] == series_id][["ds", "y"]].dropna().reset_index(drop=True)
	 splits = []
	 start = 365  # initial training window
	 while start + horizon <= len(df_local):
		 train = df_local.iloc[:start]
		 valid = df_local.iloc[start:start + horizon]
		 m = NeuralProphet(yearly_seasonality=True, weekly_seasonality=True, n_changepoints=10)
		 m.fit(train, freq="D", progress_bar=False)
		 fcst = m.predict(m.make_future_dataframe(train, periods=horizon))
		 yhat = fcst.tail(horizon)["yhat1"].to_numpy()
		 splits.append(_metrics(valid["y"].to_numpy(), yhat))
		 start += step
	 # average metrics
	 mae = float(np.mean([s["MAE"] for s in splits])) if splits else float("nan")
	 rmse = float(np.mean([s["RMSE"] for s in splits])) if splits else float("nan")
	 return {"MAE": mae, "RMSE": rmse, "folds": len(splits)}
