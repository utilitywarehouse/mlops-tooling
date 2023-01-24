# Timeseries

The Timeseries class allows you to easily create flattened datasets from a timeseries dataset, for use with models such as XGBoost and LightGBM.

An example of how to create a Timseries can be found below:

```python
example_ts = Timeseries(
    df = example_df,
    date_col='week_start_date',
    target_col='target',
    lags=[1,2,3,4,8,12,26],
    group_cols=[
        'group_a',
        'group_b'
    ],
    past_covariates=[
        'covariate_a',
        'covariate_b',
        'covariate_c',
        'covariate_d'
    ],
    covariate_lags=[1,2,3,4],
    future_covariates=[
        'covariate_a',
        'covariate_e'
    ],
    covariate_leads=[0,1],
    date_features={
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "week_of_year": "weekofyear",
        "week_of_month": "week_of_month",
        "is_first_week_of_month":"is_first_week_of_month",
        "is_last_week_of_month":"is_last_week_of_month",
    }
)

X, y, time_idx = example_ts.prepare_dataset()
```

This will take a datset with a time index, and flattened it to contain the lags specified in the lag inputs in the X dataset, and the target in the y dataset. The class forces your time column to be a datetime, and ensures your group columns are categories, which allows the output to be plugged straight into a gradient boosted model.
