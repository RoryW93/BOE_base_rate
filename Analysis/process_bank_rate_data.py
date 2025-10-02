"""
This script processes the Bank of England base rate data

# The bank rate time-series is visualised
# The histogram of the time-series bank rate data is assessed
# The PDF is parametrically estimated

# See processing_functions module for details of processing routines

Date: 24/09/25
Author: Rory White
Location: Nottingham, UK
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from processing_functions import *

# (0) Add routine to download and find BoE data from the website
input_data = "BOE" #"BOE", "ONS", "Both"
filename = check_local_data(input_data)

# (1a) pre process BoE base rate data
data, data_diff = pre_process_BoE_data()

# (1b) pre process ONS Inflation data
# pre_process_ONS_data()

# need to resample the dataset
start_data = data["Date Changed"].min()
end_data = data["Date Changed"].max()
year_diff = end_data.year - start_data.year
frequ = 4  # four data points per year (quarterly)

#time_vector = np.arange(start_data, end_data, timedelta(days=31*frequ)).astype(datetime)
#data_interp = np.interp(time_vector, data["Date Changed"].to_numpy(), data["Rate"].to_numpy())

# (1c) extract statistical summary
print("Full dataset:\n" + str(data["Rate"].describe(())))
print(f"Latest Bank Of England base rate: " + str(data["Date Changed"].iloc[-1]) + ", " + str(data["Rate"].iloc[-1]) + "%")

# (2) Visualise full time-series datasets: BoE base rate data, ONS Inflation data & rates of changes
vis_full_dataset(data)

# (3) Time-series processing and analysis
# (3a) process and analyse datasets: per year, 5 years and decade
# create time-series plots showing the distributions
# data sampling
# max
# min
# mean
# mean +/- std
# extract statistical summaries
time_diff = data["Date Changed"].diff()

annual_data = data.copy()
annual_data["year"] = annual_data["Date Changed"].dt.strftime('%Y')
annual_data = annual_data[["year","Rate"]]
annual_data_summary = pd.DataFrame()
annual_data_summary['frequ per year'] = annual_data.groupby(["year"]).count()
annual_data_summary['max per year'] = annual_data.groupby(["year"]).max()
annual_data_summary['min per year'] = annual_data.groupby(["year"]).min()
annual_data_summary['mean per year'] = annual_data.groupby(["year"]).mean()
annual_data_summary['std per year'] = annual_data.groupby(["year"]).std()

# (3b) BoE base rate and ONS Inflation data correlation and regression analysis


# (4) reference the last time the base rate data was equivalent in value (and associated inlation data value) - past dates
data_comparison = compare_data(data)

# (5) Visualise statistical distributions - including option for estimating normal PDFs (mean, sigma)
vis_stat_profile(data)
# vals, probs = estimate_PDF(data)

# (6) process and categorise base rate and inflation data rates of change (gradient profiles)
# (6a)
fig = plt.figure(5, figsize=[10, 6])
plt.subplot(2, 1, 1)
plt.plot(data["Date Changed"], data["Rate"], color='k')
plt.ylabel("Base rate [%]")

plt.subplot(2, 1, 2)
plt.plot(data_diff["Date Changed"], data_diff['Rate change'], color='k')
plt.xlabel("Time [dd, month, yy]")
plt.ylabel("Base rate difference")
plt.title("Time-series")
plt.show()

# (6b) Categorise datasets rate of change
data_diff_pos = data_diff[data_diff.iloc[:,[0,1]] > 0]
data_diff_neg = data_diff[data_diff.iloc[:,[0,1]] < 0]
data_diff_stat = data_diff[data_diff.iloc[:,[0,1]] == 0]
labels = ['Positive', 'Negative', 'Zero']

fig = plt.figure(6, figsize=[10, 6])
plt.bar(labels, [data_diff_pos["Rate change"].count(), data_diff_neg["Rate change"].count(), data_diff_stat["Rate change"].count()])
plt.ylabel("Base rate gradient change")
plt.title("Time-series")
plt.show()

fig = plt.figure(7, figsize=[10, 6])
plt.subplot(1, 2, 1)
plt.hist(data_diff_pos["Rate change"], bins=int(round(len(data_diff_pos["Rate change"]) / 20, 0)), density=True)
plt.xlabel("Base rate of change: Positive change")
plt.title("Histogram")
plt.subplot(1, 2, 2)
plt.hist(data_diff_neg["Rate change"], bins=int(round(len(data_diff_neg["Rate change"]) / 20, 0)), density=True)
plt.xlabel("Base rate of change: Negative change")
plt.title("Histogram")
plt.show()

#check the correct bin numbers

#vis_stat_profile(data_diff_pos)
#vis_stat_profile(data_diff_neg)
#vis_stat_profile(data_diff_stat)

"""
fig = plt.figure(6, figsize=[10, 6])
plt.scatter(data_diff_neg, data_diff_pos)
plt.ylabel("Base rate gradient change")
plt.title("Time-series")
plt.show()
"""
