"""
This script uses the bank of england base rate data to predict/forecast values using a LSTM recurrent Neural Network.

Process:
(1) Decompose features from the time-series data
(2) split dataset into training and test dataset
(3) Convert dataset scalar into 0-1 scale using MinMaxScaler
(4) Define the model
(5) Fit the LSTM NN model
(6) Assess the loss function related to the number of epochs
(7) Perform predictions using the train model on the final 12 data samples
(8) Convert data back to original format & visualise

LSTM hyper-parameters/functions:
Number of neurons
Activation function
Optimiser algorithm
Loss function/method

Date: 03/11/22
Author: R.White
Location: Nottingham, UK

Needs debugging and testing properly

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("bank_of_england_base_rate_history.csv", delimiter=",")


# (0) reformat data type variables:
data["Date Changed"] = pd.to_datetime(data["Date Changed"])  # , format='%Y-%m-%d')
data["Rate"] = data["Rate"].astype(float)
# flip the dataset round - start from 1975 to 2022
data = data.reindex(index=data.index[::-1])
data = data.reset_index(drop=True)

data = data.set_index(pd.DatetimeIndex(data["Date Changed"]))
data = data.drop(columns="Date Changed")

# # interpolate data to ensure it has a consistent timestamp
# data_reindexed = data.reindex(pd.date_range(start=data.index.min(),
#                                                   end=data.index.max(),
#                                                   freq='M'))
# data_reindexed.interpolate(method='linear')

# visualise the time-series
plt.figure(1)
data["Rate"].plot(figsize=(12,6))
# plt.show()

# (1) Decompose features from the time-series data
# decompose the seasonality and trend of the underlying time-series data
# results = seasonal_decompose(data['Rate'])
# results.plot()
# plt.show()

# evaluate the size of the datasets
print(len(data['Rate']))

# (2) Split dataset into training and test dataset
X = data.iloc[:]
size = int(len(X) * 0.66)
train, test = data.iloc[0:size], data.iloc[size:len(X)]
# test = [x for x in test]
# history = [x for x in train]
predictions = list()

# (3) Convert dataset scalar into 0-1 scale using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Training data based on 12 months worth of data
n_input = len(test)
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# (4) define the model
model = Sequential()  # makes sure that the NN model layers are built up sequentially one after another
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))

# LSTM description:
# 100 - 100 neurons in the layer - changed from 1000 to 100
# activation='relu' - activation function is relu function
# input_shape - shape of the array of data being fed into the layer
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# (5) Fit the LSTM NN model
epochs = 50 # parameter to train the neural network with all the training data for x number of times
model.fit(generator,epochs=epochs)

# (6) Assess the loss function related to the number of epochs
loss_per_epoch = model.history.history['loss']
plt.figure(2)
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss function')
plt.show()

# initially assess the prediction of the next data value - using the train/test set
last_train_batch = scaled_train[len(test):]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)
print('predicted=%f, expected=%f' % (model.predict(last_train_batch), scaled_test[0]))

# (7) Perform predictions using the train model on the final 12 data samples
test_predictions = []
first_eval_batch = scaled_train[-n_input:]  # take the last 12 values in the training set
current_batch = first_eval_batch.reshape((1, n_input, n_features)) # reshape data into correct input format for NN model

for i in range(len(test)):
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred)
    print('predicted=%f, expected=%f' % (current_pred, scaled_test[i]))

    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)

# (8) Convert data back to original format & visualise
true_predictions = scaler.inverse_transform(test_predictions)
plt.plot(figsize=(12,6))
plt.plot(test)
plt.plot(true_predictions)
plt.show()

# evaluate overall error metric - using root mean squared error
rmse = sqrt(mean_squared_error(test,true_predictions))
print(rmse)
