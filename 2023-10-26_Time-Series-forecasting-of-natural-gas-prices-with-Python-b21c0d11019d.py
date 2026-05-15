# Description: Short example for Time Series forecasting of natural gas prices with Python.



from math import sqrt
from prophet import Prophet
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
import datetime 
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# getting historical data for US Natural Gas prices. 
# This code calls the API and transforms the result into a DataFrame.




ticker = "NGUSD" #US natural gas
base = 'https://financialmodelingprep.com/api/v3/'
key = YOUR_KEY
target = f"{base}historical-price-full/{ticker}?apikey={key}"

df = pd.read_json(target)
df = pd.json_normalize(df['historical'])
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.to_csv('data/NGUSD data.csv')
df.head()

#Plot of asset historical closing price
df['adjClose'].plot(figsize=(10, 6), title = "Price of {} from {} to {}".format(ticker, df.index.min(), df.index.max()))

pred_end_date = datetime.datetime(2023, 7, 20)
forecast_dates = [d if d.isoweekday() in range(1, 6) else np.nan for d in pd.date_range(df.index.max(), pred_end_date)] 
intervals = len(forecast_dates)
iterations = 10000
#Preparing log returns from data
log_returns = np.log(1 + df['adjClose'].pct_change())

#Setting up drift and random component in relation to asset data
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()
daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(intervals, iterations)))

#Takes last data point as startpoint point for simulation
S0 = df['adjClose'].iloc[-1]
price_list = np.zeros_like(daily_returns)
price_list[0] = S0
#Applies Monte Carlo simulation in asset
for t in range(1, intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]

forecast_df = pd.DataFrame(price_list)

forecast_df.plot(figsize=(10,6), legend=False, title = f"{iterations} Simulated Future Paths")

# Plotting with a histogram

x = forecast_df.values[-1]
sigma = np.std(x)
mu = np.mean(x)

num_bins = 15

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1, alpha=.75)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.axvline(np.mean(x), color='r')
ax.axvline(mu+sigma*1.96, color='g', ls='--')
ax.axvline(mu-sigma*1.96, color='g', ls='--')
ax.axvline(S0)
ax.set_xlabel(f'Predicted Price on {pred_end_date}')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of {ticker}: $\mu={mu:.02f}$, $\sigma={sigma:.02f}$'.format(ticker = ticker, mu=mu, sigma=sigma))

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

forecast_df['date'] = [df.index.max()+pd.Timedelta(days=i) for i in forecast_df.index]
forecast_df.set_index('date', inplace=True)

df['Source'] = 'Actual'
forecast_df['Source'] = 'Forecast'
result = forecast_df.append(df[['adjClose', 'Source']], sort=False)

r = result.reset_index()
r = pd.melt(r, id_vars=['Source', 'date'])



# fit ARIMA model
model = ARIMA(df['adjClose'], order=(5,1,0))
model_fit = model.fit()
# summary of fit model
logger.info(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
logger.info(residuals.describe())

series = df['adjClose']
series.index = df['adjClose'].index
# split into train and test sets using time series cross-validation (last fold as test)
X = series.values
tscv = TimeSeriesSplit(n_splits=5)
idx = np.arange(len(X))
train_idx, test_idx = list(tscv.split(idx))[ -1 ]
train, test = X[train_idx], X[test_idx]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    pd.concat([predictions, yhat])
    obs = test[t]
    pd.concat([history, obs])

rmse = sqrt(mean_squared_error(test, predictions))
logger.info('Test RMSE: %.3f' % rmse)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# data must be formated for Prophet

[X, Y] = df.index, df['adjClose']
p = pd.DataFrame(Y, X)
p.reset_index(inplace=True)
p.rename(columns={'date': 'ds', 'adjClose': 'y'}, inplace=True)

m = Prophet()
m.fit(p)

future = m.make_future_dataframe(periods=5)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
