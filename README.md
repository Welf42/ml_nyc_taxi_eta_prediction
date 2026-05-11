# ML NYC Taxi ETA Prediction

Predict taxi trip duration in New York City using the Kaggle NYC Taxi Trip Duration dataset.

This is a practical lab for the **Machine Learning for Urban Mobility** theory page at `welf.dev/labs/ai-intro`. The theory page explains when ML is useful in transport planning; this repository applies those ideas to one concrete transport problem.

## Problem

Given a taxi trip's pickup time, pickup/dropoff locations, and trip attributes, estimate how long the trip will take.

The project is solving an **ETA prediction** problem: turning historical taxi trips into a model that can support more reliable travel-time estimates.

## Why ML?

Trip duration is uncertain. It depends on time of day, location, distance, traffic patterns, and operational conditions that are hard to capture with one fixed rule.

ML is useful here because the project has historical examples of completed trips and a clear target to learn from. This is not a simple deterministic calculation, a shortest-path routing problem, or a safety-critical automation system. The model estimates travel time; it does not replace routing logic or human planning judgment.

## ML Framing

- **Task type:** Regression
- **Target variable:** Trip duration
- **Why regression:** The model predicts a continuous value, not a class, future time series, or hidden group.
- **Not used here:** Classification, forecasting, and clustering.

## Python Stack

This project uses a classical machine learning stack:

- **pandas:** load, clean, and inspect trip data.
- **numpy:** numeric calculations.
- **scikit-learn:** train/test split, preprocessing, baseline models, regression models, and metrics.
- **matplotlib/seaborn:** charts for exploration and error analysis.

PyTorch and TensorFlow are not planned for this project. They are mainly used for neural networks, such as computer vision, deep time-series models, text models, or graph neural networks.

## Workflow

This project follows the same high-level ML workflow from the theory page:

1. **Problem framing:** predict taxi trip duration.
2. **Data:** use historical NYC taxi trips from Kaggle.
3. **Feature engineering:** create time, distance, and location-based features.
4. **Baseline model:** compare against a simple duration estimate before using complex models.
5. **Model:** start simple, then test stronger regression models.
6. **Evaluation:** measure error globally and by segment, such as hour, distance, or area.
7. **Decision layer:** translate model errors into transport insight, such as where ETA reliability is weak.

## Roadmap

- [ ] Load the raw Kaggle data.
- [ ] Explore trip duration, locations, time patterns, and outliers.
- [ ] Clean invalid or unrealistic trips.
- [ ] Build time, distance, and location-based features.
- [ ] Create a simple baseline model.
- [ ] Train and compare regression models.
- [ ] Evaluate errors by segment and document findings.

## Data

Raw data is not committed to this repository. Download it from Kaggle and place it under `data/raw/`.

See [`data/README.md`](data/README.md) for the expected local file structure.

## Setup

Create a local virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Register the environment as a notebook kernel:

```bash
python -m ipykernel install --user --name ml-nyc-taxi-eta --display-name "Python (.venv) - NYC Taxi ETA"
```

## Status

In progress. The repository will grow from data exploration to features, baseline models, evaluation, and interpretation.
