# https://github.com/gmonaci/ARIMA/blob/master/time-series-analysis-ARIMA.ipynb

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defaults
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('EURRUB_120101_181003.csv', engine='python', skipfooter=3)[['<DATE>', '<CLOSE>']]
# A bit of pre-processing to make it nicer
data['<DATE>'] = pd.to_datetime(data['<DATE>'], format='%Y%m%d')
data.set_index(['<DATE>'], inplace=True)

train = data['2012-01-01':'2016-12-01']
test = data['2017-01-01':'2018-01-01']

# 1 Start with a Naive Approach
dd = np.asarray(train['<CLOSE>'])
forecast = test.copy()
forecast['naive'] = dd[len(dd) - 1]

# 2 Simple Average
forecast['avg_forecast'] = train['<CLOSE>'].mean()

# 3 Moving Average
forecast['moving_avg_forecast'] = train['<CLOSE>'].rolling(60).mean().iloc[-1]

# 4 Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(np.asarray(train)).fit(smoothing_level=0.6, optimized=False)
forecast['SES'] = fit1.forecast(len(test))

# 5 Holt’s Linear Trend method
fit2 = Holt(np.asarray(train)).fit(smoothing_level=0.3, smoothing_slope=0.1)
forecast['Holt_linear'] = fit2.forecast(len(test))

# 6 Holt-Winters Method
fit3 = ExponentialSmoothing(np.asarray(train), seasonal_periods=7, trend='add', seasonal='add', ).fit()
forecast['Holt_Winter'] = fit3.forecast(len(test))

# 7 ARIMA
fit4 = SARIMAX(train, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
forecast['SARIMA'] = fit4.predict(start='2017-01-01', end='2018-01-01', dynamic=True)

plt.figure(figsize=(16, 8))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast['naive'], label='naive')
plt.plot(forecast['avg_forecast'], label='avg_forecast')
plt.plot(forecast['moving_avg_forecast'], label='moving_avg_forecast')
plt.plot(forecast['SES'], label='SES')
plt.plot(forecast['Holt_linear'], label='Holt_linear')
plt.plot(forecast['Holt_Winter'], label='Holt_Winter')
plt.plot(forecast['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = []
rms.append(np.sqrt(mean_squared_error(test, forecast['naive'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['avg_forecast'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['moving_avg_forecast'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['SES'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['Holt_linear'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['Holt_Winter'])))
rms.append(np.sqrt(mean_squared_error(test, forecast['SARIMA'])))
print(rms)
