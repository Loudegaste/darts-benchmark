# Darts-benchmark
Darts-benchmark is a set of scripts used to compare the performance of different Darts models on custom datasets. It includes **Auto-ML functionnalities** whith Optuna hyperparameter gridsearch, as well as other utils to compare and tune models. A google colab showcase of the tools developed is available at https://colab.research.google.com/github/Loudegaste/darts-benchmark/blob/main/benchmark.ipynb 





Long time horizon performance with 10 minutes of Optuna tuning per model:

| dataset        |   AutoARIMA |   FFT |   LightGBMModel |   LinearRegressionModel |   NBEATSModel |   NHiTSModel |   NLinearModel |   NaiveSeasonal |   Prophet |   TCNModel |
|:---------------|------------:|------:|----------------:|------------------------:|--------------:|-------------:|---------------:|----------------:|----------:|-----------:|
| Air passengers |       0.304 | 0.276 |           0.514 |                   0.094 |         0.313 |        0.241 |          0.362 |           0.377 |     0.285 |      1.213 |
| ETTh1          |       0.402 | 0.397 |           0.235 |                   0.286 |         0.359 |        0.37  |          0.386 |           0.423 |     0.416 |      0.47  |
| ExchangeRate   |       0.329 | 0.511 |           0.228 |                   0.289 |         0.36  |        0.342 |          0.301 |           0.33  |     0.706 |      1.143 |
| GasRateCO2     |       0.261 | 0.659 |           0.261 |                   0.26  |         0.308 |        0.354 |          0.388 |           0.362 |     0.702 |      0.335 |
| Sunspots       |       1.089 | 0.807 |           0.481 |                   0.651 |         0.829 |        0.642 |          0.582 |           1.358 |     1.061 |      0.663 |
| USGasoline     |       0.592 | 0.539 |           0.399 |                   0.385 |         0.443 |        0.389 |          0.486 |           0.724 |     0.612 |      0.494 |
| Weather        |       0.537 | 0.502 |           0.07  |                   0.078 |         0.106 |        0.091 |          0.076 |           0.602 |     0.499 |      0.079 |


Shorte time horizon performance with 10 minutes of Optuna tuning per model:
| dataset        |   AutoARIMA |   FFT |   LightGBMModel |   LinearRegressionModel |   NBEATSModel |   NHiTSModel |   NLinearModel |   NaiveSeasonal |   Prophet |   TCNModel |
|:---------------|------------:|------:|----------------:|------------------------:|--------------:|-------------:|---------------:|----------------:|----------:|-----------:|
| Air passengers |       0.468 | 0.299 |           0.52  |                   0.112 |         0.356 |        0.225 |          0.338 |           0.61  |     0.299 |      0.895 |
| ETTh1          |       0.123 | 0.446 |           0.076 |                   0.078 |         0.114 |        0.114 |          0.111 |           0.125 |     0.335 |      0.115 |
| ExchangeRate   |       0.049 | 0.275 |           0.119 |                   0.048 |         0.067 |        0.071 |          0.058 |           0.05  |     0.55  |      0.054 |
| GasRateCO2     |       0.261 | 0.659 |           0.261 |                   0.26  |         0.308 |        0.354 |          0.388 |           0.362 |     0.702 |      0.335 |
| Sunspots       |       0.377 | 0.794 |           0.354 |                   0.357 |         0.412 |        0.372 |          0.352 |           0.381 |     0.873 |      0.389 |
| USGasoline     |       0.385 | 0.437 |           0.251 |                   0.306 |         0.323 |        0.347 |          0.372 |           0.399 |     0.345 |      0.423 |
| Weather        |       0.105 | 0.303 |           0.043 |                   0.059 |         0.099 |        0.116 |          0.078 |           0.202 |     0.37  |      0.083 |