"""
Based on initial assessment in process_bank_rate_data.py, the parameters for an ARIMA model are estimated.
This script uses the estimated ARIMA hyper-parameters to build a model and assess the performance

Initial model hyper-parameter estimate: ARIMA(p=1, d=1,q= 2)

Date: 13/09/22
Author: R.White
Location: Nottingham, UK

TODO: Model and forecast time-history
Follow:
https://machinelearningmastery.com/make-sample-forecasts-arima-python/
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
import math

# statistical modelling/computations
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

data = pd.read_csv("bank_of_england_base_rate_history.csv", delimiter=",")


# reformat data type variables:
data["Date Changed"] = pd.to_datetime(data["Date Changed"])  # , format='%Y-%m-%d')
data["Rate"] = data["Rate"].astype(float)
# flip the dataset round - start from 1975 to 2022
data = data.reindex(index=data.index[::-1])
data = data.reset_index(drop=True)

# TODO: Need to interpolate and create a new interpolated time-series with consistent time sampling

# data_inverted = DataFrame
# for ii in range(len(data)):
#     data_inverted.append(data["Date Changed"][ii], ignore_index=True)

# Build initial model and assess
# arima_model = ARIMA(data["Rate"], order=(1,1,2)) # original: ARIMA(p=1, d=1,q= 2)
# model_fit = arima_model.fit()


# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# # density plot of residuals
# residuals.plot(kind='kde')
# plt.show()
# # summary stats of residuals
# print(residuals.describe())

# output = model_fit.forecast()
# plt.figure()
# plt.plot(output, color='k')
# plt.plot(data["Rate"], color='b')
# plt.show()

# split into train and test sets
X = data.iloc[:,1]
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
test = [x for x in test]
history = [x for x in train]
predictions = list()

# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(1, 2, 2))  # 0, 2, 2
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()