# Data

Raw data is not included in this repository.

Download the NYC Taxi Trip Duration dataset from Kaggle after accepting the competition rules:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

Place the files here:

```text
data/
`-- raw/
    |-- train.csv
    |-- test.csv
    `-- sample_submission.csv
```

`data/raw/` is ignored by git.

## Dataset Overview

Each row represents one NYC taxi trip.

Main variables:

- `pickup_datetime`: trip start time.
- `dropoff_datetime`: trip end time, available in the training data.
- `pickup_longitude`, `pickup_latitude`: pickup location.
- `dropoff_longitude`, `dropoff_latitude`: dropoff location.
- `passenger_count`: number of passengers.
- `trip_duration`: target variable, measured in seconds.

## Train and Test Files

- `train.csv` includes `trip_duration`, so it is used for exploration, training, and evaluation.
- `test.csv` does not include `trip_duration`, so it is used for prediction after a model has been trained.
- `sample_submission.csv` shows the expected Kaggle submission format.

This project predicts `trip_duration` from trip timing, location, distance, and related features. Deeper exploration, data quality checks, and visualizations belong in notebooks or analysis scripts.
