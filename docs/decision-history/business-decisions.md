# Business Decisions

Durable project decisions that affect business logic, product behavior, data interpretation, or implementation policy.

## Decision Log

### 2026-07-18 - Remove live weather-assisted forecasting

- **Decision:** Remove `POST /v1/weather-assisted-forecasts` and its Open-Meteo integration. The public API accepts only complete UCI-compatible feature inputs.
- **Context:** The artifact is evaluated on historical 2011-2012 Capital Bikeshare data and does not include the current-demand, capacity, event, or retraining data needed to support credible live Washington, DC forecasts.
- **Rationale:** External weather resolution was peripheral to the project’s primary demonstration and risked implying that a live provider made the historical artifact valid for current demand forecasting.
- **Implications:** Preserve deterministic feature-input inference and the historical-evaluation framing. Do not reintroduce a live forecast endpoint without a validated current-data and retraining pipeline.
- **Status:** Active.
- **References:** README.md; bike_demand_api/routes.py; bike_demand_api/service.py.

### 2026-07-17 - Condition weather drift baselines by month and hour

- **Decision:** Evaluate weather-feature PSI against training distributions conditioned on month and hour, weighted to the incoming batch's month/hour composition. Do not use raw `month` or `hour` mix as a drift signal.
- **Context:** A global weather baseline would treat normal summer-versus-winter and day-versus-night changes as input drift for this bike-demand model.
- **Rationale:** Month × hour captures the most material expected weather seasonality without the sparse, over-segmented cohorts that a finer workday/hour design would create.
- **Implications:** Weather drift is a conditional feature-quality diagnostic, not a claim that any seasonal distribution change is anomalous. Future persisted monitoring should preserve the same cohort definition when aggregating windows.
- **Status:** Active.
- **References:** bike_demand_api/ml.py; artifacts/approved/artifact_manifest.json; POST /v1/forecasts.

### 2026-07-16 - Position the API as a historical model deployment demo

- **Decision:** Present the project as a reproducible historical ML deployment and governance demonstration, not as a validated current Washington DC demand-forecasting product.
- **Context:** The public Capital Bikeshare dataset covers 2011-2012.
- **Rationale:** The repository's purpose is to demonstrate sound modeling, artifact governance, API inference, and external feature mapping. It does not include recent observed demand, capacity, event data, or scheduled retraining required for credible current operational forecasting.
- **Implications:** Preserve the chronological historical evaluation framing. Do not add stale temporal-trend features or demand-lag features without the corresponding historical-data pipeline.
- **Status:** Active.
- **References:** README.md; bike_demand_api/static/index.html; POST /v1/forecasts.
