import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def compare_models(time_series, periods=12):

    # Make sure input is a DataFrame
    if isinstance(time_series, pd.Series):
        time_series = time_series.reset_index()

    time_series = time_series.copy()

    # Take last column as target
    target_column = time_series.columns[-1]

    # Convert target to numeric
    values = pd.to_numeric(
        time_series[target_column],
        errors="coerce"
    ).dropna()

    # Need enough data
    if len(values) < 20:
        raise ValueError(
            "Not enough data for model comparison."
        )

    # Train-test split
    train_size = int(len(values) * 0.8)

    train = values.iloc[:train_size]
    test = values.iloc[train_size:]

    # ==========================================
    # ARIMA
    # ==========================================

    arima_model = ARIMA(
        train,
        order=(1, 1, 1)
    )

    arima_fit = arima_model.fit()

    arima_pred = arima_fit.forecast(
        steps=len(test)
    )

    arima_mae = mean_absolute_error(
        test,
        arima_pred
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test,
            arima_pred
        )
    )

    arima_mape = np.mean(
        np.abs(
            (test.values - arima_pred.values)
            / test.values
        )
    ) * 100

    # ==========================================
    # EXPONENTIAL SMOOTHING
    # ==========================================

    try:

        es_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=None
        )

        es_fit = es_model.fit()

        es_pred = es_fit.forecast(
            len(test)
        )

        es_mae = mean_absolute_error(
            test,
            es_pred
        )

        es_rmse = np.sqrt(
            mean_squared_error(
                test,
                es_pred
            )
        )

        es_mape = np.mean(
            np.abs(
                (test.values - es_pred.values)
                / test.values
            )
        ) * 100

    except Exception:

        es_mae = np.nan
        es_rmse = np.nan
        es_mape = np.nan

    # ==========================================
    # RESULTS
    # ==========================================

    results = pd.DataFrame({
        "Model": [
            "ARIMA",
            "Exponential Smoothing"
        ],
        "MAE": [
            arima_mae,
            es_mae
        ],
        "RMSE": [
            arima_rmse,
            es_rmse
        ],
        "MAPE": [
            arima_mape,
            es_mape
        ]
    })

    # Best model
    valid_results = results.dropna(
        subset=["MAPE"]
    )

    best_row = valid_results.loc[
        valid_results["MAPE"].idxmin()
    ]

    best_model = best_row["Model"]
    best_mape = best_row["MAPE"]

    return results, best_model, best_mape
