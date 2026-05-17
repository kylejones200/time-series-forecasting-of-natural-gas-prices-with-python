"""Generated from Jupyter notebook: Shell and Brent Crude regression

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import datetime
import json
import time
from datetime import datetime, timedelta
from urllib.request import urlopen

import certifi
import investpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is not stationary. Differencing required.")
    else:
        print(f"{name} is stationary.")


def download_data_safely(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed. Waiting 10 seconds before retry...")
            time.sleep(10)


def get_historical_price(symbol, key, from_date, to_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&to={to_date}&apikey={key}"
    data = get_jsonparsed_data(url)
    if "historical" in data:
        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()
        return df
    else:
        return None


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def get_shell_and_brent_crude_data() -> None:
    shell = yf.Ticker("SHEL")

    brent = yf.Ticker("BNO")

    end_date = datetime.datetime.now()

    start_date = end_date - datetime.timedelta(days=3 * 365)

    shell_data = shell.history(start=start_date, end=end_date)

    brent_data = brent.history(start=start_date, end=end_date)

    df_merged = pd.DataFrame(
        {"Shell_Close": shell_data["Close"], "Brent_Close": brent_data["Close"]}
    )

    df_merged = df_merged.resample("D").ffill()

    df_merged["Shell_Returns"] = df_merged["Shell_Close"].pct_change()

    df_merged["Brent_Returns"] = df_merged["Brent_Close"].pct_change()

    df_merged = df_merged.dropna()

    df_merged.to_csv("shell_brent_returns.csv")

    print("Correlation between Shell and Brent prices:")

    print(df_merged[["Shell_Close", "Brent_Close"]].corr())

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df_merged[["Shell_Close", "Brent_Close"]].corr(),
        annot=True,
        cmap="coolwarm",
        center=0,
    )

    plt.title("Correlation Heatmap: Shell vs Brent")

    plt.show()

    X = df_merged["Brent_Close"]

    Y = df_merged["Shell_Close"]

    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    print("\nRegression Results:")

    print(model.summary())

    print("\nGranger Causality Test Results:")

    print("Testing if Brent crude prices Granger-cause Shell stock prices")

    gc_test = grangercausalitytests(df_merged[["Shell_Close", "Brent_Close"]], maxlag=5)

    plt.figure(figsize=(12, 6))

    plt.plot(df_merged.index, df_merged["Shell_Close"], label="Shell Stock Price")

    plt.plot(df_merged.index, df_merged["Brent_Close"], label="Brent Crude Price")

    plt.title("Shell Stock Price vs Brent Crude Price")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.show()

    plt.figure(figsize=(10, 6))

    plt.scatter(df_merged["Brent_Close"], df_merged["Shell_Close"], alpha=0.5)

    plt.plot(df_merged["Brent_Close"], model.predict(X), color="red", linewidth=2)

    plt.xlabel("Brent Crude Price")

    plt.ylabel("Shell Stock Price")

    plt.title("Shell Stock Price vs Brent Crude Price with Regression Line")

    plt.show()


def function_to_safely_download_data_with_retry_logi() -> None:
    end_date = datetime.datetime.now()

    start_date = end_date - datetime.timedelta(days=3 * 365)

    tickers = "SHEL BNO"

    df = yf.download(tickers, start=start_date, end=end_date)["Close"]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["Shell_Close", "Brent_Close"]

    df = df.dropna()

    df["Shell_Returns"] = df["Shell_Close"].pct_change()

    df["Brent_Returns"] = df["Brent_Close"].pct_change()

    df = df.dropna()

    print("Correlation between Shell and Brent prices:")

    correlation = df[["Shell_Close", "Brent_Close"]].corr()

    print(correlation)

    plt.figure(figsize=(10, 8))

    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)

    plt.title("Correlation Heatmap: Shell vs Brent")

    plt.show()

    X = df["Brent_Close"]

    Y = df["Shell_Close"]

    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    print("\nRegression Results:")

    print(model.summary())

    print("\nGranger Causality Test Results:")

    print("Testing if Brent crude prices Granger-cause Shell stock prices")

    gc_test = grangercausalitytests(df[["Shell_Close", "Brent_Close"]], maxlag=5)

    plt.figure(figsize=(12, 6))

    df[["Shell_Close", "Brent_Close"]].plot()

    plt.title("Shell Stock Price vs Brent Crude Price")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.show()

    plt.figure(figsize=(10, 6))

    plt.scatter(df["Brent_Close"], df["Shell_Close"], alpha=0.5)

    plt.plot(df["Brent_Close"], model.predict(X), color="red", linewidth=2)

    plt.xlabel("Brent Crude Price")

    plt.ylabel("Shell Stock Price")

    plt.title("Shell Stock Price vs Brent Crude Price with Regression Line")

    plt.show()

    print("\nKey Statistics:")

    print(f"R-squared: {model.rsquared:.4f}")

    print(f"Correlation coefficient: {correlation.iloc[0, 1]:.4f}")

    print("\nInterpretation:")

    print("If the p-values in the Granger Causality test are < 0.05, ")

    print("we can say that Brent crude prices Granger-cause Shell stock prices.")


def get_shell_data() -> None:
    shell = investpy.get_stock_historical_data(
        stock="XOM",
        country="United States",
        from_date="01/01/2020",
        to_date="01/01/2023",
    )

    brent = investpy.get_commodity_historical_data(
        commodity="Brent Oil", from_date="01/01/2020", to_date="01/01/2023"
    )


def notebook_step_004() -> None:
    key = "<your_key_here>"


def for_python_3_0_and_later() -> None:
    try:
        from urllib.request import urlopen
    except ImportError:
        pass

    url = f"https://financialmodelingprep.com/api/v3/search?query=AA&apikey={key}"

    print(get_jsonparsed_data(url))


def notebook_step_006() -> None:
    url


def set_your_api_key() -> None:
    end_date = datetime.now()

    start_date = end_date - timedelta(days=3 * 365)

    from_date = start_date.strftime("%Y-%m-%d")

    to_date = end_date.strftime("%Y-%m-%d")

    shell_data = get_historical_price("SHEL.L", key, from_date, to_date)

    brent_data = get_historical_price("BRENT", key, from_date, to_date)

    data = pd.DataFrame({"Shell": shell_data["close"], "Brent": brent_data["close"]})

    data = data.dropna()

    plt.figure(figsize=(12, 6))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Date")

    ax1.set_ylabel("Shell Stock Price (GBP)", color="tab:blue")

    ax1.plot(data.index, data["Shell"], color="tab:blue")

    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Brent Crude Price (USD)", color="tab:orange")

    ax2.plot(data.index, data["Brent"], color="tab:orange")

    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Shell Stock Price vs Brent Crude Price")

    plt.show()

    print("\nStationarity Tests:")

    check_stationarity(data["Shell"], "Shell")

    check_stationarity(data["Brent"], "Brent")

    data["Shell_Returns"] = data["Shell"].pct_change()

    data["Brent_Returns"] = data["Brent"].pct_change()

    data = data.dropna()

    correlation = data[["Shell", "Brent"]].corr()

    print("\nCorrelation between Shell and Brent prices:")

    print(correlation)

    print("\nGranger Causality Test Results:")

    print("Testing if Brent crude prices Granger-cause Shell stock prices")

    gc_test = grangercausalitytests(data[["Shell_Returns", "Brent_Returns"]], maxlag=5)

    print("\nKey Statistics:")

    print(f"Shell daily volatility: {data['Shell_Returns'].std() * 100:.2f}%")

    print(f"Brent daily volatility: {data['Brent_Returns'].std() * 100:.2f}%")

    print(f"Price correlation: {correlation.iloc[0, 1]:.4f}")

    print(
        f"Returns correlation: {data[['Shell_Returns', 'Brent_Returns']].corr().iloc[0, 1]:.4f}"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Shell_Returns"], label="Shell Returns", alpha=0.7)

    plt.plot(data.index, data["Brent_Returns"], label="Brent Returns", alpha=0.7)

    plt.title("Daily Returns")

    plt.xlabel("Date")

    plt.ylabel("Returns")

    plt.legend()

    plt.grid(True)

    plt.show()

    plt.figure(figsize=(10, 6))

    plt.scatter(data["Brent_Returns"], data["Shell_Returns"], alpha=0.5)

    plt.xlabel("Brent Crude Returns")

    plt.ylabel("Shell Stock Returns")

    plt.title("Shell Returns vs Brent Returns")

    plt.grid(True)

    plt.show()


def main() -> None:
    get_shell_and_brent_crude_data()
    function_to_safely_download_data_with_retry_logi()
    get_shell_data()
    notebook_step_004()
    for_python_3_0_and_later()
    notebook_step_006()
    set_your_api_key()


if __name__ == "__main__":
    main()
