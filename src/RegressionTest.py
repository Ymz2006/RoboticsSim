import tensorflow as tf
from keras import Sequential
from keras import layers
import numpy as np

# Define the model
model = Sequential()

# Add a single dense layer with 1 neuron and 4 inputs
model.add(layers.Dense(1, input_shape=(4,)))

# Compile the model
model.compile(optimizer='adam', loss='mae')  # Mean Squared Error loss for regression

# Generate some dummy data for training
# Let's assume a + b + c + d = y
np.random.seed(42)
X = np.random.rand(2000, 4)  # 1000 samples, 4 features (a, b, c, d)
y = np.sum(X, axis=1)/0.034  # Target is the sum of a, b, c, d

# Train the model
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Test the model with a new example
test_input = np.array([[1, 2, 3, 4]])  # Example input
predicted_output = model.predict(test_input)
print(f"Predicted output for {test_input} is: {predicted_output}")