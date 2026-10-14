import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def prepare_time_series(df, date_column, target_column):

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.dropna()

    data = data.sort_values(date_column)

    # Combine multiple records having the same date
    time_series = (
        data.groupby(date_column)[target_column]
        .sum()
        .reset_index()
    )

    return time_series


def forecast_arima(
    time_series,
    date_column,
    target_column,
    periods=12
):

    # Create time series
    series = time_series.set_index(
        date_column
    )[target_column]

    # Make sure index is datetime
    series.index = pd.to_datetime(series.index)

    # Remove duplicate dates
    series = series.groupby(
        series.index
    ).sum()

    # ARIMA model
    model = ARIMA(
        series,
        order=(1, 1, 1)
    )

    model_fit = model.fit()

    # Future predictions
    predictions = model_fit.forecast(
        steps=periods
    )

    # Weekly frequency
    last_date = series.index[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=periods,
        freq="W"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": predictions.values
    })

    return forecast_df