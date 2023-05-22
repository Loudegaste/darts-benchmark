import json
import logging
import os
import warnings
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from darts_benchmark.model_evaluation import evaluate_model
from darts_benchmark.optuna_search import optuna_search, Dataset
from darts_benchmark.param_space import FIXED_PARAMS

from darts import TimeSeries
from darts.metrics import mae


def convert_to_ts(ds: TimeSeries):
    """
    Converts enventual numerical index to datetime index
    (necessary for compatibility down the line)
    """
    return TimeSeries.from_times_and_values(
        pd.to_datetime(ds.time_index), ds.all_values(), columns=ds.components
    )


def experiment(
    list_datasets: List[Dataset],
    models,
    grid_search=False,
    time_budget=120,
    experiment_dir=None,
    forecast_horizon=1,
    repeat=3,
    metric=mae,
    num_test_points=20,
    silent_search=False,
    scale_model=True,
):
    """
    Takes a list of datasets
    """

    split = 0.8
    if experiment_dir:
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)
    else:
        temp_dir = TemporaryDirectory()
        experiment_dir = str(temp_dir.name)

    # we use dict to handle multiple entries of the same dataset/model
    path_results = os.path.join(experiment_dir, "results.json")
    if os.path.isfile(path_results):
        results = json.load(open(path_results))
    else:
        results = dict()

    for dataset in list_datasets:
        fh_corrected = forecast_horizon
        if forecast_horizon < 1:
            fh_corrected = int(forecast_horizon * (1 - split) * len(dataset.series))
        print(f"Using forecast horizon of length {fh_corrected}")

        for model_class in models:

            stride = int((1 - split) * len(dataset.series) / num_test_points)
            stride = max(stride, 1)

            if (
                dataset.name in results
                and model_class.__name__ in results[dataset.name]
            ):
                continue

            model_params = FIXED_PARAMS[model_class.__name__](
                series=dataset.series.split_after(split)[0], forecast_horizon=fh_corrected
            )
            if silent_search:
                silence_prompt()
            if grid_search and time_budget:
                model_params = optuna_search(
                    model_class,
                    fixed_params=model_params,
                    dataset=dataset,
                    time_budget=time_budget,
                    optuna_dir=os.path.join(experiment_dir, "optuna"),
                    forecast_horizon=fh_corrected,
                    metric=metric,
                    stride=stride,
                    split=split,
                    scale_model=scale_model,
                )

            output = evaluate_model(
                model_class,
                series=dataset.series,
                past_covariates=dataset.past_covariates,
                future_covariates=dataset.future_covariates,
                model_params=model_params,
                split=split,
                forecast_horizon=fh_corrected,
                repeat=repeat,
                metric=metric,
                stride=stride,
                scale_model=scale_model,
            )

            print("#####################################################")
            print(dataset.name, model_class.__name__, output)

            if dataset.name not in results:
                results[dataset.name] = dict()
            results[dataset.name][model_class.__name__] = output

            # save results
            json.dump(results, open(path_results, "w"))
    results = format_output(results)
    return results


def format_output(results: Dict[str, Dict[str, float]]):
    """Turns a nested dictionnary with (column name, model name, metric value) into a pandas dataframe"""
    results = [  # type: ignore
        (dataset, model, metric)
        for dataset, model_dict in results.items()
        for model, metric in model_dict.items()
    ]
    df = pd.DataFrame(results, columns=["dataset", "model", "metric"])
    df = (
        df.groupby(["dataset", "model"], as_index=False)
        .first()
        .pivot(index="dataset", columns="model", values="metric")
    )
    return df


def illustrate(models, dataset, split=0.8, forecast_horizon=1, 
               grid_search=False, time_budget=60, metric=mae, 
               num_test_points=20, scale_model=True, silent_search=False):

    fh_corrected = forecast_horizon

    if forecast_horizon < 1:
        fh_corrected = int(forecast_horizon * (1 - split) * len(dataset.series))
    print(f"Using forecast horizon of length {fh_corrected}")
    stride = int((1 - split) * len(dataset.series) / num_test_points)
    stride = max(stride, 1)

    limit = int(len(dataset.series) * split * 0.95)
    target = dataset.series[limit:]
    target.plot()
    if silent_search:
        silence_prompt()
    for model_class in models:
        if grid_search:
            model_params = optuna_search(
                model_class,
                dataset=dataset,
                time_budget=time_budget,
                forecast_horizon=fh_corrected,
                metric=metric,
                scale_model=scale_model,
            )
        else:
            model_params = FIXED_PARAMS[model_class.__name__](**dataset._asdict())

        _, output = evaluate_model(  # type: ignore
            model_class,
            series=dataset.series,
            past_covariates=dataset.past_covariates,
            future_covariates=dataset.future_covariates,
            model_params=model_params,
            split=split,
            forecast_horizon=fh_corrected,
            get_output_sample=True,
            scale_model=scale_model,
        )
        output.plot(label=f"{model_class.__name__} \n"
                          f"{metric.__name__}={metric(output, target):.2f}")


def silence_prompt():
    # ray, optuna and pytorch: all are very verbose, and ray tune
    # forwards the output for multiple processes in parallel
    # so we need to silence warnings to make the prompt readable

    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    warnings.filterwarnings("ignore")
    for logger in loggers:
        logger.setLevel(logging.ERROR)
