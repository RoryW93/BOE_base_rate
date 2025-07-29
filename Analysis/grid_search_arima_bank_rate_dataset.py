"""
This script uses the grid search method to evaluate the correct hyperparamter set for the ARIMA model: (p,d,q)

Following tutorial:
https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

This is applied to the current bank rate dataset
"""

import warnings
import math
import pandas as pd
import os
from grid_search_functions import evaluate_arima_model, evaluate_models, parser

# load in bank rate csv data
data = pd.read_csv("bank_of_england_base_rate_history.csv", delimiter=",")

# reformat data type variables:
data["Date Changed"] = pd.to_datetime(data["Date Changed"]) # , format='%Y-%m-%d')
data["Rate"] = data["Rate"].astype(float)
# flip and change the index <<< up to here
data = data.iloc[::-1,:]
# data = data.reset_index(drop=True)

# evaluate parameters
p_values = range(0, 3) #[0, 1, 2, 3, 4, 5, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)

# turn advisory warnings off
warnings.filterwarnings("ignore")
evaluate_models(data["Rate"], p_values, d_values, q_values)

