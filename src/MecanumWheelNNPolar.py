import numpy as np
import tensorflow as tf
import os
import pandas as pd
import keras
from keras import layers
from keras import ops

# df = pd.read_csv('mecanum_polar_rot.csv')
#
#
#
# inputs = df.iloc[:, :4]  # Features (input)
# outputs = df.iloc[:, 4:]  # Target (output)
#
# print(inputs.head())
# print (outputs.head())
# inputs = np.array(inputs)
# outputs = np.array(outputs)
#
# model = keras.Sequential()
# model.add(layers.Dense(1, input_shape=(4,)))
#
#
# model.compile(optimizer='adam', loss='mse')
# model.fit(inputs, outputs, epochs = 100, batch_size = 32, verbose = 1)


df = pd.read_csv('mecanum_polar_speed.csv')

inputs = df.iloc[:, :4]  # Features (input)
outputs = df.iloc[:, 4:]  # Target (output)


model2 = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)),  # Hidden layer with 16 neurons
    layers.Dense(1)  # Output layer (no activation for regression)
])
model2.compile(optimizer='adam', loss='mse')
model2.fit(inputs, outputs, epochs = 100, batch_size = 32, verbose = 1)


