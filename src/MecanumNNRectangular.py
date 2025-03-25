import numpy as np
import tensorflow as tf
import os
import pandas as pd
import keras
from keras import layers
from keras import ops

df = pd.read_csv('mecanum_data.csv')



inputs = df.iloc[:, :4]  # Features (input)
outputs = df.iloc[:, 4:]  # Target (output)

print(inputs.head())
print (outputs.head())
inputs = np.array(inputs)
outputs = np.array(outputs)

outputs_mean,outputs_std = outputs.mean(), outputs.std()  # Compute mean and standard deviation

outputs_norm = (outputs - outputs_mean) / outputs_std  # Standardize y
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),  # Input layer
    layers.Dense(32, activation='relu'),  # Hidden layer 1
    layers.Dense(16, activation='relu'),  # Hidden layer 2
    layers.Dense(3)  # Output layer (no activation for regression)
])

model.compile(optimizer='adam', loss='mae')
model.fit(inputs, outputs, epochs = 100, batch_size = 32, verbose = 1)


