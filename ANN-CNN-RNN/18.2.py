import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Create sequential data (Predicting the next number)
# Shape: (samples, time_steps, features) -> (4 samples, 3 steps, 1 feature)
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=np.float32)

y = np.array([[4], [5], [6], [7]], dtype=np.float32)

# 2. Build the structural layout of the RNN
model = Sequential([
    SimpleRNN(units=5, input_shape=(3, 1), activation='relu'), # Hidden loop layer
    Dense(units=1)                                             # Output layer
])

# 3. Compile the network (assign optimizer and loss metric)
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the model
model.fit(X, y, epochs=1000, verbose=0)

# 5. Test with a completely new sequence
test_input = np.array([[[5], [6], [7]]], dtype=np.float32)
print("Predicted next number:", model.predict(test_input).round())
