import datetime
import logging
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA


def data_must_be_formated_for_prophet(df) -> None:
    [X, Y] = (df.index, df["adjClose"])
    p = pd.DataFrame(Y, X)
    p.reset_index(inplace=True)
    p.rename(columns={"date": "ds", "adjClose": "y"}, inplace=True)
    m = Prophet()
    m.fit(p)
    future = m.make_future_dataframe(periods=5)
    future.tail()
    forecast = m.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
    m.plot(forecast)


def fit_arima_model(df, forecast_df, logger) -> None:
    forecast_df["date"] = [
        df.index.max() + pd.Timedelta(days=i) for i in forecast_df.index
    ]
    forecast_df.set_index("date", inplace=True)
    df["Source"] = "Actual"
    forecast_df["Source"] = "Forecast"
    result = forecast_df.append(df[["adjClose", "Source"]], sort=False)
    r = result.reset_index()
    r = pd.melt(r, id_vars=["Source", "date"])
    model = ARIMA(df["adjClose"], order=(5, 1, 0))
    model_fit = model.fit()
    logger.info(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind="kde")
    plt.show()


def main_step_002(
    S0,
    daily_returns,
    df,
    intervals,
    iterations,
    logger,
    pred_end_date,
    price_list,
    ticker,
) -> None:
    for t in range(1, intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    forecast_df = pd.DataFrame(price_list)
    forecast_df.plot(
        figsize=(10, 6), legend=False, title=f"{iterations} Simulated Future Paths"
    )
    x = forecast_df.values[-1]
    sigma = np.std(x)
    mu = np.mean(x)
    num_bins = 15
    the_histogram_of_the_data(S0, mu, num_bins, pred_end_date, sigma, ticker, x)
    fit_arima_model(df, forecast_df, logger)
    summary_stats_of_residuals(df, logger, residuals)
    data_must_be_formated_for_prophet(df)


def summary_stats_of_residuals(df, logger, residuals) -> None:
    logger.info(residuals.describe())
    series = df["adjClose"]
    series.index = df["adjClose"].index
    X = series.values
    tscv = TimeSeriesSplit(n_splits=5)
    idx = np.arange(len(X))
    train_idx, test_idx = list(tscv.split(idx))[-1]
    train, test = (X[train_idx], X[test_idx])
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    rmse = sqrt(mean_squared_error(test, predictions))
    logger.info(f"Test RMSE: {rmse:.3f}")
    plt.plot(test)
    plt.plot(predictions, color="red")
    plt.show()


def the_histogram_of_the_data(
    S0, mu, num_bins, pred_end_date, sigma, ticker, x
) -> None:
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x, num_bins, density=1, alpha=0.75)
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)
    ax.plot(bins, y, "--")
    ax.axvline(np.mean(x), color="r")
    ax.axvline(mu + sigma * 1.96, color="g", ls="--")
    ax.axvline(mu - sigma * 1.96, color="g", ls="--")
    ax.axvline(S0)
    ax.set_xlabel(f"Predicted Price on {pred_end_date}")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Histogram of {ticker}: $\\mu={mu:.02f}$, $\\sigma={sigma:.02f}$")
    fig.tight_layout()
    plt.show()


def seed() -> None:
    global df, logger, ticker
    np.random.seed(42)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    ticker = "NGUSD"
    base = "https://financialmodelingprep.com/api/v3/"
    key = YOUR_KEY
    target = f"{base}historical-price-full/{ticker}?apikey={key}"
    df = pd.read_json(target)
    df = pd.json_normalize(df["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.to_csv("data/NGUSD data.csv")


def head() -> None:
    global df, logger, ticker
    df.head()
    df["adjClose"].plot(
        figsize=(10, 6),
        title=f"Price of {ticker} from {df.index.min()} to {df.index.max()}",
    )
    pred_end_date = datetime.datetime(2023, 7, 20)
    forecast_dates = [
        d if d.isoweekday() in range(1, 6) else np.nan
        for d in pd.date_range(df.index.max(), pred_end_date)
    ]
    intervals = len(forecast_dates)
    iterations = 10000
    log_returns = np.log(1 + df["adjClose"].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - 0.5 * var
    stdev = log_returns.std()
    daily_returns = np.exp(
        drift + stdev * norm.ppf(np.random.rand(intervals, iterations))
    )
    S0 = df["adjClose"].iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    main_step_002(
        S0,
        daily_returns,
        df,
        intervals,
        iterations,
        logger,
        pred_end_date,
        price_list,
        ticker,
    )


def main() -> None:
    seed()
    head()


if __name__ == "__main__":
    main()
