"""
This file implements wrappers around backtest to have the same input structure for all models/all datasets
"""
import logging
import random
from typing import Callable, Dict, Optional, Tuple, Union
from darts_benchmark.param_space import FIXED_PARAMS
import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.metrics import mae
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

# Add some options for deep learning models
early_stopper = EarlyStopping("train_loss", min_delta=0.001, patience=3, verbose=True)
PL_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "accelerator": "cpu",
    "callbacks": [early_stopper],
}


def set_randommness(seed=0):
    # set randomness for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate_model(
    model_class,
    series: TimeSeries,
    model_params: Union[Dict, None] = None,
    metric: Callable[[TimeSeries, TimeSeries], float] = mae,
    split: float = 0.85,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    stride: int = 1,
    forecast_horizon=1,
    repeat=1,
    get_output_sample=False,
    scale_data=True,
) -> Union[float, Tuple[float, TimeSeries]]:
    """
    Evaluates a model on a given dataset. It will retrain the model for a number of times equal to `repeat` and return
    the average metric value. If `get_output_sample` is True, it will also return the output sample of the model.
        
    Parameters:
    -----------
    model_class: ForecastingModel
        The class of the model to be evaluated.
    series: TimeSeries
        The dataset to be used for evaluation.
    model_params: Dict
        The parameters of the model to be evaluated.
    metric: Callable[[TimeSeries, TimeSeries], float]
        The metric to be used for evaluation.
    split: float
        Portion of the dataset to be used for training.
    past_covariates: TimeSeries
    future_covariates: TimeSeries
    stride: int
        Interval between subsequent inference points.
    forecast_horizon: int
        The forecast horizon of the model
    repeat: int
        Number of times to retrain the model.
    get_output_sample: bool
        Whether to return a prediction exemple of the model.
    scale_data: bool
        Whether to scale the model inputs.
    """

    set_randommness()
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    scaler = Scaler(StandardScaler())
    
    if model_params is None:
        print(f"No model params provided for model {model_class.__name__}," 
              "using the ones from the param_space.FIXED_PARAMS")
        model_params = FIXED_PARAMS[model_class.__name__](
            series=series,
            forecast_horizon=forecast_horizon,
            past_covariates=past_covariates,
            future_covariates=future_covariates
            )      

    # Standardizes inputs to have the same entry point for all models and datasets
    if issubclass(model_class, TorchForecastingModel):
        model_params["pl_trainer_kwargs"] = PL_TRAINER_KWARGS

    if scale_data:
        if past_covariates:
            past_covariates = scaler.fit_transform(past_covariates)
        if future_covariates:
            future_covariates = scaler.fit_transform(future_covariates)
        series = scaler.fit_transform(series)

    # now we performe the evaluation
    model_instance = model_class(**model_params)
    is_local_model = isinstance(model_instance, LocalForecastingModel)
    retrain = True if is_local_model else retrain_func

    function_args = {
        "series": series,
        "retrain": retrain,
        "start": split,
        "stride": stride,
        "metric": metric,
        "forecast_horizon": forecast_horizon,
        "last_points_only": True,
    }
    if metric is not None:
        function_args["metric"] = metric
    if model_instance.supports_past_covariates and not model_params.get(
        "shared_weights"
    ):
        function_args["past_covariates"] = past_covariates
    if model_instance.supports_future_covariates and not model_params.get(
        "shared_weights"
    ):
        function_args["future_covariates"] = future_covariates

    # performing evaluation
    mean_losse = np.mean([model_instance.backtest(**function_args) for _ in range(repeat)])
    
    if not get_output_sample:
        return mean_losse
    else:
        # historical_forecasts doesn't accept metric as an argument
        if "metric" in function_args:
            del function_args["metric"]
        hist_forecast = model_instance.historical_forecasts(**function_args)
        if repeat == 1:
            # if we only have one repeat, we might as well output the error of the single run
            mean_losse = metric(series, hist_forecast)
        if scale_data:
            hist_forecast = scaler.inverse_transform(hist_forecast)
        return mean_losse, hist_forecast


def retrain_func(counter, pred_time, train_series, past_covariates, future_covariates):
    """A retrain function telling the model to retrain
    only once at the start of the historical forecast prediction"""
    return counter == 0
