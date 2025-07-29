"""
This script processes the Bank of England base rate data - starting from 1975 to present (07/09/2022)

# The bank rate time-series is visualised
# The histogram of the time-series bank rate data is assessed
# The PDF is parametrically estimated
# The first- and second-order derivatives of the bank rate time-series data are estimated
# The auto-correlation function of the time-series (including the first- and second-order derivatives)
# The dickey-fuller test is computed to assess the stationarity of the time-series data and its derivatives

Date: 07/09/22
Author: Rory White
Location: Nottingham, UK

# updated 12/09/22
# updated 03/11/22
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from scipy import signal
from datetime import timedelta
from datetime import date

# statistical modelling/computations
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# placeholder - add web scraping functionality using beautiful soup
# https://realpython.com/beautiful-soup-web-scraper-python/#reasons-for-web-scraping
# https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp

# automatically wants to save as: Bank Rate history and data  Bank of England Database.csv

data = pd.read_csv("bank_of_england_base_rate_history.csv", delimiter=",")

# ---------------------------------------------------------------------------------------------------------------------#
# Perform initial processing on dataset
# ---------------------------------------------------------------------------------------------------------------------#
# reformat data type variables:
data["Date Changed"] = pd.to_datetime(data["Date Changed"])  # , format='%Y-%m-%d')
data["Rate"] = data["Rate"].astype(float)
# flip the dataset round - start from 1975 to latest date
data = data.reindex(index=data.index[::-1])
data = data.reset_index(drop=True)

# check if timestamp is consistent or requires resampling via interpolation
data["Date Changed"].diff().plot()
plt.show()

# extract statistical summary
print("Full dataset:\n" + str(data["Rate"].describe(())))
print(f"Latest Bank Of England base rate: " + str(data["Date Changed"].iloc[-1]) + ", " + str(data["Rate"].iloc[-1]) + "%")

# reference the last time the base rate was equivalent in value - past timedate
data_comparison = data[data["Rate"] == data["Rate"].iloc[-1]]
data_comparison = data_comparison.reset_index(drop=True)
print("The last time the base rate was " + str(data_comparison["Rate"][1]) + "%:\n" + str(
    data_comparison["Date Changed"]))

# identify timeframe/span of data
time_span = data["Date Changed"].max() - data["Date Changed"].min()

# compute the moving average of the BOE base rate
data["Rate MA"] = data["Rate"].rolling(window=3).mean()

# Evaluate distribution of data within +/-1 s.d
data_1sd = data[(data["Rate"].mean() - data["Rate"].std() < data["Rate"]) & (
        data["Rate"] < data["Rate"].mean() + data["Rate"].std())]

# extract data from 2005 - lower interest data comparison
# year_diff = (date.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

data_from2004 = data[data["Date Changed"].dt.year >= 2004]
print("\n\n" + str(data_from2004["Date Changed"].iloc[-1]) + " dataset:\n" + str(data_from2004["Rate"].describe(())))

# add functionality for evaluating the statistics every three years

# ---------------------------------------------------------------------------------------------------------------------#
# Estimate the statistical distribution of the data
# ---------------------------------------------------------------------------------------------------------------------#
# https://machinelearningmastery.com/probability-density-estimation/
# Estimate the PDF from the data:
# Parametric: Assuming an approximate normal distribution
dist = norm(data["Rate"].mean(), data["Rate"].std())
values = [value for value in range(int(data["Rate"].min()), int(data["Rate"].max()))]
# values = [value for value in range(1, 15)]
probabilities = [dist.pdf(value) for value in values]

# Non-parametric: Kernel density estimation
model = KernelDensity(bandwidth=2, kernel='gaussian')
sample = data["Date Changed"].to_numpy().reshape((len(data), 1))
model.fit(sample)
values_np = np.asarray([value for value in range(int(data["Rate"].min()), int(data["Rate"].max()))])
values_np = values_np.reshape((len(values_np), 1))
probabilities_np = model.score_samples(values_np)
probabilities_np = np.exp(probabilities_np)

# ---------------------------------------------------------------------------------------------------------------------#
# Visualise the time-series data
# ---------------------------------------------------------------------------------------------------------------------#
fig = plt.figure(1, figsize=[10, 6])
plt.plot(data["Date Changed"], data["Rate"], color='k')
# plt.plot(data["Date Changed"], data["Rate MA"], color='r')
plt.scatter(data_1sd["Date Changed"], data_1sd["Rate"], color='m')
plt.scatter(data_comparison["Date Changed"], data_comparison["Rate"], color='b')
plt.plot(data["Date Changed"], data_comparison["Rate"].mode()[0]*np.ones(np.shape(data["Date Changed"])), color='b', linestyle='--')
plt.xlabel("Time [dd, month, yy]")
plt.ylabel("Base rate [%]")
plt.title("Time-series")
plt.legend(["All data", u"\u00B1 $\sigma$", "Current comparison: " + str(data["Rate"].iloc[-1]) +"%", "Comparison line"])  # r"+/- $\sigma$")

# ---------------------------------------------------------------------------------------------------------------------#
# Visualise the time-series data and statistical distribution
# ---------------------------------------------------------------------------------------------------------------------#
fig = plt.figure(2, figsize=[10, 6])
plt.subplot(1, 3, 1)
plt.plot(data["Date Changed"], data["Rate"], color='k')
# plt.plot(data["Date Changed"], data["Rate MA"], color='r')
plt.scatter(data_1sd["Date Changed"], data_1sd["Rate"], color='m')
plt.scatter(data_comparison["Date Changed"], data_comparison["Rate"], color='b')
plt.xlabel("Time [dd, month, yy]")
plt.ylabel("Base rate")
plt.title("Time-series")
plt.legend(["All data", u"\u00B1 $\sigma$", "Current comparison: " + str(data["Rate"].iloc[-1]) +"%"])  # r"+/- $\sigma$")

plt.subplot(1, 3, 2)
plt.hist(data["Rate"], bins=int(round(len(data["Rate"]) / 20, 0)), density=True)
plt.plot(values, probabilities, color='r')
plt.plot(values_np, probabilities_np, color='c')
plt.xlabel("Base rate")
plt.title("Histogram")

plt.subplot(1, 3, 3)
plt.boxplot(data["Rate"])
plt.title("Box and Whisker")
plt.show()

# Bayesian optimisation modelling
# https://machinelearningmastery.com/what-is-bayesian-optimization/

# https://www.projectpro.io/article/how-to-build-arima-model-in-python/544#:~:text=Model%20in%20Python%3F-,ARIMA%20Model%2D%20Complete%20Guide%20to%20Time%20Series%20Forecasting%20in%20Python,data%20to%20predict%20future%20values.
# forecast bank rate - ARIMA/SARIMA
# use Scipy for auto-correlation assessment
corr = signal.correlate(data["Rate"], data["Rate"], mode='full', method='auto')

# ---------------------------------------------------------------------------------------------------------------------#
# visualise the auto-correlation function
# ---------------------------------------------------------------------------------------------------------------------#

plt.figure(3, figsize=[10, 6])
plt.subplot(1, 2, 1)
plt.plot(data["Date Changed"], data["Rate"], color='b')
plt.xlabel("Time [dd, month, yy]")
plt.ylabel("Base rate")
plt.title("Auto-correlation")
ax = plt.figure(3, figsize=[10, 6]).add_subplot(1, 2, 2)
# plt.plot(data["Date Changed"], corr, color='b')
plot_acf(data["Rate"], ax=ax)
ax.set_ylim([0, 1.25])
plt.show()

fig = plt.figure(4, figsize=[12, 6])
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(data["Date Changed"], data["Rate"], color='b')
ax1.set_xlabel("Time [dd, month, yy]")
ax1.set_ylabel("Base rate")

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(data["Date Changed"], data["Rate"].diff(), color='b')
ax2.set_xlabel("Time [dd, month, yy]")
ax2.set_ylabel("Base rate derivative")

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(data["Date Changed"], data["Rate"].diff().diff(), color='b')
ax3.set_xlabel("Time [dd, month, yy]")
ax3.set_ylabel("Base rate derivative")

ax4 = fig.add_subplot(2, 3, 4)
plot_acf(data["Rate"], ax=ax4)
ax4.set_ylabel("Base rate autocorrelation")

ax5 = fig.add_subplot(2, 3, 5)
plot_acf(data["Rate"].diff().dropna(), ax=ax5)
ax5.set_ylabel("Base rate autocorrelation")

ax6 = fig.add_subplot(2, 3, 6)
plot_acf(data["Rate"].diff().diff().dropna(), ax=ax6)
ax6.set_ylabel("Base rate autocorrelation")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------#
# Summarise augmented dickey-fuller test results
# ---------------------------------------------------------------------------------------------------------------------#
print('p-value: ', adfuller(data["Rate"])[1])
print('p-value: ', adfuller(data["Rate"].diff().dropna())[1])
print('p-value: ', adfuller(data["Rate"].diff().diff().dropna())[1])

# Conclusion:
# Assessing the time-series data -
# (1) Data follows some approximate signs of a normal distribution
# (2) Lags from the auto-correlation function demonstrates that the first sample is key
# (3) The dickey-fuller test signifies the first order derivative displays stationarity

# Order of differencing, d = 2 (start with 1)
# Moving average parameter, q = 2
# p = 1
# Thus, our final ARIMA model can be initially defined as ARIMA(p=1, d=1,q= 2)

# analyse the data for the current year
data_CY = data[pd.DatetimeIndex(data["Date Changed"]).year == date.today().year]
