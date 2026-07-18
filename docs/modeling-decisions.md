# Modeling Decisions

This document explains the modeling choices in the Bike Demand Forecast API. They are intentionally proportionate to a historical public-data deployment demonstration, not universal defaults for every forecasting problem.

## Promotion metric: MAE

The candidate model is promoted only when its validation mean absolute error (MAE) is at least 1% lower than the seasonal-median baseline's MAE. MAE expresses the typical absolute demand error in rides and is less dominated by occasional demand spikes than RMSE. RMSE is still reported in the artifact manifest and API metadata because larger misses remain useful operational context.

## Seasonal-median baseline

The baseline groups training observations by hour, month, and working-day status, then predicts each group's median demand. Those fields capture the dominant recurring structure in this dataset: intraday demand shape, annual seasonality, and the workday versus non-workday distinction. The median provides a robust, interpretable benchmark that is fitted only on the training period.

## One-percent promotion threshold

A nonzero threshold prevents a candidate from replacing the baseline because of a negligible validation difference. The 1% value is a simple policy for this demonstration: meaningful enough to reject noise-level improvements, but modest enough that a clearly better candidate can be promoted. In a production system, the threshold would be selected from the cost of error, model-operating cost, and uncertainty in the evaluation result.

## Prediction intervals from validation residuals

The API derives its interval radius from the 90th percentile of absolute residuals on the validation period for the approved model. Applying that radius symmetrically around each prediction produces an empirical residual-based interval; the lower bound is clipped at zero because demand cannot be negative. Using validation residuals avoids calibrating the interval on the same observations used to fit the model. It is not a conditional uncertainty model or a guaranteed confidence interval.

## Weather drift conditioned by month and hour

Temperature, humidity, windspeed, and weather condition naturally vary by season and time of day. Comparing a July afternoon request batch to one global training-weather distribution would incorrectly flag ordinary seasonality as drift. The artifact therefore stores month-by-hour weather reference distributions. At request time, expected proportions are weighted to the batch's observed month/hour mix before PSI is calculated.

## Deliberately limited tuning

The project compares one seeded XGBoost candidate against a transparent baseline rather than running an extensive hyperparameter search. That keeps the artifact reproducible, makes the promotion rule easy to inspect, and avoids repeatedly optimizing against the single validation period. The project is designed to demonstrate defensible training-to-serving mechanics, not claim the highest possible score on this dataset.
