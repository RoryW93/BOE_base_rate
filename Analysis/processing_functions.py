"""
This module contains functions to process the Bank of England base rate data

Processing routines:
- The bank rate time-series is visualised
- The histogram of the time-series bank rate data is assessed
- The PDF is parametrically estimated
- The first- and second-order derivatives of the bank rate time-series data are estimated
- The auto-correlation function of the time-series (including the first- and second-order derivatives)
- The dickey-fuller test is computed to assess the stationarity of the time-series data and its derivatives

Date: 14/06/25
Author: Rory White
Location: Nottingham, UK
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


def pre_process_data():
    """
    Loads in bank of england base rate data and pre-processes into user-friendly format

    :return: data - [Date Changed, Rate]
    """

    # load bank of england base rate time series - pandas dataframe
    data = pd.read_csv("bank_of_england_base_rate_history.csv", delimiter=",")

    # convert data into: Date - datemine Rate - float
    data["Date Changed"] = pd.to_datetime(data["Date Changed"])  # , format='%Y-%m-%d')
    data["Rate"] = data["Rate"].astype(float)

    # flip the dataset round - start from 1975 to latest date
    data = data.reindex(index=data.index[::-1])
    data = data.reset_index(drop=True)

    return data


def compare_data(data):
    """
    Evaluate each time the base rate time series was equivalent in value to latest datapoint
    :param data:
    :return: data_comparison - [Date Changed, Rate]
    """

    data_comparison = data[data["Rate"] == data["Rate"].iloc[-1]]
    data_comparison = data_comparison.reset_index(drop=True)

    print("The last time the base rate was " + str(data_comparison["Rate"][1]) + "%:\n" +
          str(data_comparison["Date Changed"])
          )

    return data_comparison


def filter_data_2004(data):
    """
    Filter data timeseries from 2004 - allow better post-processing and analysis of last 20 years
    :param data:
    :return:
    """
    data_from2004 = data[data["Date Changed"].dt.year >= 2004]

    print("\n\n" +
          str(data_from2004["Date Changed"].iloc[-1]) +
          " dataset:\n" +
          str(data_from2004["Rate"].describe())
          )

    return data_from2004


def estimate_PDF(data):
    """
    Estimate the probability density function/estimation
    Assumes an approximate normal distribution - parametric function
        # https://machinelearningmastery.com/probability-density-estimation/
    # Estimate the PDF from the data:
    # Parametric: Assuming an approximate normal distribution
    :param data:
    :return:
    """

    dist = norm(data["Rate"].mean(), data["Rate"].std())
    values = [value for value in range(int(data["Rate"].min()), int(data["Rate"].max()))]
    probabilities = [dist.pdf(value) for value in values]

    # Non-parametric: Kernel density estimation
    model = KernelDensity(bandwidth=2, kernel='gaussian')
    sample = data["Date Changed"].to_numpy().reshape((len(data), 1))
    model.fit(sample)
    values_np = np.asarray([value for value in range(int(data["Rate"].min()), int(data["Rate"].max()))])
    values_np = values_np.reshape((len(values_np), 1))
    probabilities_np = model.score_samples(values_np)
    probabilities_np = np.exp(probabilities_np)

    return values, probabilities


def vis_full_dataset(data):

    # extract equivalent data
    data_comparison = compare_data(data)

    # Evaluate distribution of data within +/-1 s.d
    data_1sd = data[(data["Rate"].mean() - data["Rate"].std() < data["Rate"]) &
                    (data["Rate"] < data["Rate"].mean() + data["Rate"].std())
                    ]

    fig = plt.figure(1, figsize=[10, 6])
    plt.plot(data["Date Changed"], data["Rate"], color='k')
    plt.scatter(data_1sd["Date Changed"], data_1sd["Rate"], color='m')
    plt.scatter(data_comparison["Date Changed"], data_comparison["Rate"], color='b')
    plt.plot(data["Date Changed"], data_comparison["Rate"].mode()[0] * np.ones(np.shape(data["Date Changed"])),
             color='b', linestyle='--')
    plt.xlabel("Time [dd, month, yy]")
    plt.ylabel("Base rate [%]")
    plt.title("Time-series")
    plt.legend(["All data", u"\u00B1 $\sigma$", "Current comparison: " + str(data["Rate"].iloc[-1]) + "%",
                "Comparison line"])  # r"+/- $\sigma$")
    plt.show()


def vis_stat_profile(data):
    """
    Visualise statistical profile/distribution of the base rate data
    - Time-series response
    - Histogram & PDF
    - Box and whisker
    :param data:
    :return:
    """

    # extract equivalent data
    data_comparison = compare_data(data)

    # Evaluate distribution of data within +/-1 s.d
    data_1sd = data[(data["Rate"].mean() - data["Rate"].std() < data["Rate"]) &
                    (data["Rate"] < data["Rate"].mean() + data["Rate"].std())
                    ]

    # Evaluate probability distribution
    values, probabilities = estimate_PDF(data)

    fig = plt.figure(1, figsize=[10, 6])
    plt.subplot(1, 3, 1)
    plt.plot(data["Date Changed"], data["Rate"], color='k')
    # plt.plot(data["Date Changed"], data["Rate MA"], color='r')
    plt.scatter(data_1sd["Date Changed"], data_1sd["Rate"], color='m')
    plt.scatter(data_comparison["Date Changed"], data_comparison["Rate"], color='b')
    plt.xlabel("Time [dd, month, yy]")
    plt.ylabel("Base rate")
    plt.title("Time-series")
    plt.legend(
        ["All data", u"\u00B1 $\sigma$", "Current comparison: " + str(data["Rate"].iloc[-1]) + "%"])  # r"+/- $\sigma$")

    plt.subplot(1, 3, 2)
    plt.hist(data["Rate"], bins=int(round(len(data["Rate"]) / 20, 0)), density=True)
    plt.plot(values, probabilities, color='r')
    plt.xlabel("Base rate")
    plt.title("Histogram")

    plt.subplot(1, 3, 3)
    plt.boxplot(data["Rate"])
    plt.title("Box and Whisker")
    plt.show()


def vis_acf(data):

    """
    Visualise the auto-correlation of the base rate data
    Determine stationarity

    :param data:
    :return:
    """

    # use Scipy for auto-correlation assessment
    corr = signal.correlate(data["Rate"], data["Rate"], mode='full', method='auto')

    # -----------------------------------------------------------------------------------------------------------------#
    # visualise the auto-correlation function
    # -----------------------------------------------------------------------------------------------------------------#
    plt.figure(1, figsize=[10, 6])
    plt.subplot(1, 2, 1)
    plt.plot(data["Date Changed"], data["Rate"], color='b')
    plt.xlabel("Time [dd, month, yy]")
    plt.ylabel("Base rate")
    plt.title("Auto-correlation")
    ax = plt.figure(1, figsize=[10, 6]).add_subplot(1, 2, 2)
    # plt.plot(data["Date Changed"], corr, color='b')
    plot_acf(data["Rate"], ax=ax)
    ax.set_ylim([0, 1.25])
    plt.show()

    fig = plt.figure(2, figsize=[12, 6])
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


def compute_DF_test(data):
    """
    Compute a Dickey-Fuller test

    The Dickey-Fuller test (DFT) is a statistical test used to determine whether a time series data has a unit root,
    which indicates non-stationarity.

    :param data:
    :return:
    """

    print('p-value: ', adfuller(data["Rate"])[1])
    print('p-value: ', adfuller(data["Rate"].diff().dropna())[1])
    print('p-value: ', adfuller(data["Rate"].diff().diff().dropna())[1])


def cluster_analysis(data):
    """
    KMeans clustering analysis using
    To evaluate the correct number of clusters - perform the elbow criterion:
    # see below for stack overflow elbow technique criterion
    # https://stackoverflow.com/questions/43784903/scikit-k-means-clustering-performance-measure
    :param data:
    :return:
    """

    # compute elbow technique for k means cluster
    sse = {}  # use the sum of squared error as a quantitative error metric
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        sse[k] = kmeans.inertia_

    kmeans_diff = np.diff(list(sse.values()))

    # plot k-means clustering elbow technique
    plt.figure(1, figsize=[10, 6])
    plt.subplot(2, 1, 1)
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.title('k-means clustering elbow technique')
    plt.subplot(2, 1, 2)
    plt.plot(kmeans_diff)
    # plt.title('Gradient of elbow')
    plt.show()

    # process kmeans elbow technique data - evaluate approximate correct number of cluster labels

    # evaluate most suitable model
    model = KMeans(n_clusters=3)
    model.fit(data)
    labels = model.predict(data)

    return labels
