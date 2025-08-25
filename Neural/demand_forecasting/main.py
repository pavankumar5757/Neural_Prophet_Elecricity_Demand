import click
from pathlib import Path
from src.ingest import load_and_clean_demand_data
from src.holidays_events import build_holiday_events_df, build_lockdown_events_df
from src.weather import fetch_regional_weather, engineer_weather_features
from src.merge import merge_all
from src.models import train_baseline_model, train_advanced_model, train_global_model
from src.validation import rolling_origin_cv
from src.tuning import grid_search_tuning
from config import DATA_DIR, DEFAULT_DEMAND_CSV

@click.group()
def cli():
	 pass

@cli.command()
@click.option("--file", "file_path", type=click.Path(exists=True), default=str(DEFAULT_DEMAND_CSV))
def ingest(file_path: str):
	 df_wide, df_long = load_and_clean_demand_data(Path(file_path))
	 click.echo(f"Ingested: wide={df_wide.shape}, long={df_long.shape}")

@cli.command()
@click.option("--start", type=str, default="2016-01-01")
@click.option("--end", type=str, default=None)
@click.option("--regions", multiple=True, default=())
def weather(start: str, end: str, regions: tuple[str, ...]):
	 df_raw = fetch_regional_weather(start, end, list(regions) if regions else None)
	 df_feat = engineer_weather_features(df_raw)
	 click.echo(f"Weather: raw={df_raw.shape}, features={df_feat.shape}")

@cli.command()
@click.option("--start", type=str, default="2016-01-01")
@click.option("--end", type=str, default=None)
@click.option("--file", "file_path", type=click.Path(exists=True), default=str(DEFAULT_DEMAND_CSV))
def merge(file_path: str, start: str, end: str):
	 df = merge_all(Path(file_path), start, end)
	 click.echo(f"Merged: {df.shape}")

@cli.command()
@click.option("--id", "series_id", type=str, default="NR_Demand")
@click.option("--advanced/--no-advanced", default=False)
@click.option("--global-model", is_flag=True, default=False)
def train(series_id: str, advanced: bool, global_model: bool):
	 if global_model:
		 model, forecast = train_global_model()
	 else:
		 if advanced:
			 model, forecast = train_advanced_model(series_id)
		 else:
			 model, forecast = train_baseline_model(series_id)
	 click.echo(f"Trained. Forecast rows={len(forecast)}")

@cli.command()
@click.option("--id", "series_id", type=str, default="NR_Demand")
@click.option("--horizon", type=int, default=30)
@click.option("--step", type=int, default=30)
def cv(series_id: str, horizon: int, step: int):
	 metrics = rolling_origin_cv(series_id, horizon=horizon, step=step)
	 click.echo(metrics)

@cli.command()
@click.option("--id", "series_id", type=str, default="NR_Demand")
@click.option("--max-trials", type=int, default=20)
def tune(series_id: str, max_trials: int):
	 best = grid_search_tuning(series_id, max_trials=max_trials)
	 click.echo(best)

@cli.command()
@click.option("--phase", type=click.Choice(["all_baseline"], case_sensitive=False), required=True)
@click.option("--file", "file_path", type=click.Path(exists=True), default=str(DEFAULT_DEMAND_CSV))
@click.option("--start", type=str, default="2016-01-01")
@click.option("--end", type=str, default=None)
def run(phase: str, file_path: str, start: str, end: str):
	 if phase == "all_baseline":
		 load_and_clean_demand_data(Path(file_path))
		 build_holiday_events_df()
		 build_lockdown_events_df()
		 fetch_regional_weather(start, end, None)
		 engineer_weather_features()
		 merge_all(Path(file_path), start, end)
		 train_baseline_model("NR_Demand")
		 click.echo("Completed baseline pipeline.")

if __name__ == "__main__":
	 cli()
