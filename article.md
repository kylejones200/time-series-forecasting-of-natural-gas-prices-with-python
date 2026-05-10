# Time Series forecasting of natural gas prices with Python A common task in finance is forecasting values. There are several
methods for creating forecasts such as ARIMA, Bayesian Structural Time...

### Time Series forecasting of natural gas prices with Python
#### A common task in finance is forecasting values. There are several methods for creating forecasts such as ARIMA, Bayesian Structural Time Series, and simulation. Monte Carlo is a simulation technique based on generating random walks over a period of time.
In this tutorial, we will:

1.  [fetch Natural Gas prices]
2.  [Simulate future price movements using Monte Carlo]
3.  [Plot the simulated future paths]
4.  [Calculate future values using ARIMA and Bayesian Timeseries]

We start by importing the modules we will need. We are going to access the price of natural gas from Financial Modeling Prep, a web API. You need an API key from them to pull the data.



We pull the data and save it as an CSV so we can use it later. Let's plot the data so we can see how the price has changed over time.



<figcaption>Price of Natural Gas in the US from July 2018 to July 2023</figcaption>


### Geometric Brownian Motion
We want to simualte future values for the natrual gas prices based on the historic values. To make this more relaistic, we constrain the amount the price can change in a period to the mean of the historic price change with some random shock. We assume the log of the returns (percent changes) are normally distributed. We also assume the market is efficient.

The formula for the change in price between periods is the price of the stock in t0 multiplied by the expected drift (average change in price) plus an exogenous shock.


<figcaption>Formulat for Brownian Motion</figcaption>




We run the simulation 10,000 times. We could run it more but this is good enough for now. Let's plot the end values. These are the expected values for the price of natural gas in 14 periods (in this case 14 days).



This histogram shows us the mean expected value and the range of expected values in 2 weeks. Based on this we see that it is possible to have a price of \$5 but that is unlikely.

### More advanced graphing
The data frame we created for the simulations held each run as a separate column. This is called `wide` format which is useful for some analytics tasks, but is not optimal for visualization. We use a technique called `pd.melt` to collages the 1000 columns into three. This new format, called `long`, is aligned with the principles of `tidy data`.

### Plotly
Plotly is a high level library for interactive visualization. It is a "declarative" library where we say what we want, not how to produce what we want. Notice that the code for this visualization is much smaller than the code for Matplotlib, and yet, the result is much better.


### Autoregressive integrated moving average (Arima)
ARIMA is a simple way to forecast future values using statistics. StatsModels has a nice ARIMA tool built in and we can use that to predict future values.





The red line is the predicted values and the blue like (barely visible) are the actual values.

### Prophet
Prophet is a Bayesian time series method implemented by Facebook. It allows us to use prior info to predict future values. The graph shows the actual values (dots) and the predicted range (light blue).
