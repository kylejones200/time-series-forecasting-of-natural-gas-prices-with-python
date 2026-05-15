"""Generated from Jupyter notebook: Shell and Brent Crude regression

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests

# Get Shell and Brent crude data
shell = yf.Ticker("SHEL")
# Using Brent crude ETF as a proxy for Brent prices
brent = yf.Ticker("BNO")

# Get historical data for the last 5 years
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=3 * 365)

shell_data = shell.history(start=start_date, end=end_date)
brent_data = brent.history(start=start_date, end=end_date)

# Create a dataframe with both Shell and Brent prices
df_merged = pd.DataFrame(
    {"Shell_Close": shell_data["Close"], "Brent_Close": brent_data["Close"]}
)

# Resample to daily frequency and forward fill any missing values
df_merged = df_merged.resample("D").ffill()

# Calculate returns
df_merged["Shell_Returns"] = df_merged["Shell_Close"].pct_change()
df_merged["Brent_Returns"] = df_merged["Brent_Close"].pct_change()

# Remove any missing values
df_merged = df_merged.dropna()
df_merged.to_csv("shell_brent_returns.csv")

# Correlation analysis
print("Correlation between Shell and Brent prices:")
print(df_merged[["Shell_Close", "Brent_Close"]].corr())

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_merged[["Shell_Close", "Brent_Close"]].corr(),
    annot=True,
    cmap="coolwarm",
    center=0,
)
plt.title("Correlation Heatmap: Shell vs Brent")
plt.show()

# Linear Regression
X = df_merged["Brent_Close"]
Y = df_merged["Shell_Close"]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print("\nRegression Results:")
print(model.summary())

# Granger Causality Test
print("\nGranger Causality Test Results:")
print("Testing if Brent crude prices Granger-cause Shell stock prices")
gc_test = grangercausalitytests(df_merged[["Shell_Close", "Brent_Close"]], maxlag=5)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df_merged.index, df_merged["Shell_Close"], label="Shell Stock Price")
plt.plot(df_merged.index, df_merged["Brent_Close"], label="Brent Crude Price")
plt.title("Shell Stock Price vs Brent Crude Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df_merged["Brent_Close"], df_merged["Shell_Close"], alpha=0.5)
plt.plot(df_merged["Brent_Close"], model.predict(X), color="red", linewidth=2)
plt.xlabel("Brent Crude Price")
plt.ylabel("Shell Stock Price")
plt.title("Shell Stock Price vs Brent Crude Price with Regression Line")
plt.show()


# --- code cell ---

import datetime
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests


# Function to safely download data with retry logic
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


# Set date range
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=3 * 365)  # 3 years of data

# Download data with single download call to minimize API requests
tickers = "SHEL BNO"
df = yf.download(tickers, start=start_date, end=end_date)["Close"]

# Clean column names if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["Shell_Close", "Brent_Close"]

# Basic data cleaning
df = df.dropna()

# Calculate returns
df["Shell_Returns"] = df["Shell_Close"].pct_change()
df["Brent_Returns"] = df["Brent_Close"].pct_change()
df = df.dropna()

# Correlation analysis
print("Correlation between Shell and Brent prices:")
correlation = df[["Shell_Close", "Brent_Close"]].corr()
print(correlation)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: Shell vs Brent")
plt.show()

# Linear Regression
X = df["Brent_Close"]
Y = df["Shell_Close"]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print("\nRegression Results:")
print(model.summary())

# Granger Causality Test
print("\nGranger Causality Test Results:")
print("Testing if Brent crude prices Granger-cause Shell stock prices")
gc_test = grangercausalitytests(df[["Shell_Close", "Brent_Close"]], maxlag=5)

# Visualizations
plt.figure(figsize=(12, 6))
df[["Shell_Close", "Brent_Close"]].plot()
plt.title("Shell Stock Price vs Brent Crude Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df["Brent_Close"], df["Shell_Close"], alpha=0.5)
plt.plot(df["Brent_Close"], model.predict(X), color="red", linewidth=2)
plt.xlabel("Brent Crude Price")
plt.ylabel("Shell Stock Price")
plt.title("Shell Stock Price vs Brent Crude Price with Regression Line")
plt.show()

# Print key statistics
print("\nKey Statistics:")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Correlation coefficient: {correlation.iloc[0, 1]:.4f}")

# Interpretation of Granger Causality
print("\nInterpretation:")
print("If the p-values in the Granger Causality test are < 0.05, ")
print("we can say that Brent crude prices Granger-cause Shell stock prices.")


# --- code cell ---

import investpy
import pandas as pd

# Get Shell data
shell = investpy.get_stock_historical_data(
    stock="XOM", country="United States", from_date="01/01/2020", to_date="01/01/2023"
)

# Get Brent data
brent = investpy.get_commodity_historical_data(
    commodity="Brent Oil", from_date="01/01/2020", to_date="01/01/2023"
)


# --- code cell ---

key = "<your_key_here>"


# --- code cell ---

#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import json

import certifi


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


url = f"https://financialmodelingprep.com/api/v3/search?query=AA&apikey={key}"
print(get_jsonparsed_data(url))


# --- code cell ---

url


# --- code cell ---

import json
from datetime import datetime, timedelta
from urllib.request import urlopen

import certifi
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def get_historical_price(symbol, key, from_date, to_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&to={to_date}&apikey={key}"
    data = get_jsonparsed_data(url)
    if "historical" in data:
        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()  # Sort by date
        return df
    else:
        return None


# Set your API key


# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=3 * 365)  # 3 years of data
from_date = start_date.strftime("%Y-%m-%d")
to_date = end_date.strftime("%Y-%m-%d")

# Get Shell data (LSE listing)
shell_data = get_historical_price("SHEL.L", key, from_date, to_date)

# Get Brent Crude data
brent_data = get_historical_price("BRENT", key, from_date, to_date)

# Create combined dataframe
data = pd.DataFrame({"Shell": shell_data["close"], "Brent": brent_data["close"]})

# Remove any missing values
data = data.dropna()

# Plot the original data
plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Shell stock price
ax1.set_xlabel("Date")
ax1.set_ylabel("Shell Stock Price (GBP)", color="tab:blue")
ax1.plot(data.index, data["Shell"], color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Create second y-axis for Brent price
ax2 = ax1.twinx()
ax2.set_ylabel("Brent Crude Price (USD)", color="tab:orange")
ax2.plot(data.index, data["Brent"], color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title("Shell Stock Price vs Brent Crude Price")
plt.show()


# Function to test stationarity
def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is not stationary. Differencing required.")
    else:
        print(f"{name} is stationary.")


# Test stationarity
print("\nStationarity Tests:")
check_stationarity(data["Shell"], "Shell")
check_stationarity(data["Brent"], "Brent")

# Calculate returns
data["Shell_Returns"] = data["Shell"].pct_change()
data["Brent_Returns"] = data["Brent"].pct_change()
data = data.dropna()

# Correlation analysis
correlation = data[["Shell", "Brent"]].corr()
print("\nCorrelation between Shell and Brent prices:")
print(correlation)

# Granger Causality Test
print("\nGranger Causality Test Results:")
print("Testing if Brent crude prices Granger-cause Shell stock prices")
gc_test = grangercausalitytests(data[["Shell_Returns", "Brent_Returns"]], maxlag=5)

# Calculate key statistics
print("\nKey Statistics:")
print(f"Shell daily volatility: {data['Shell_Returns'].std() * 100:.2f}%")
print(f"Brent daily volatility: {data['Brent_Returns'].std() * 100:.2f}%")
print(f"Price correlation: {correlation.iloc[0, 1]:.4f}")
print(
    f"Returns correlation: {data[['Shell_Returns', 'Brent_Returns']].corr().iloc[0, 1]:.4f}"
)

# Plot returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Shell_Returns"], label="Shell Returns", alpha=0.7)
plt.plot(data.index, data["Brent_Returns"], label="Brent Returns", alpha=0.7)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot of returns
plt.figure(figsize=(10, 6))
plt.scatter(data["Brent_Returns"], data["Shell_Returns"], alpha=0.5)
plt.xlabel("Brent Crude Returns")
plt.ylabel("Shell Stock Returns")
plt.title("Shell Returns vs Brent Returns")
plt.grid(True)
plt.show()
