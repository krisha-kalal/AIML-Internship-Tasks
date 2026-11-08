import numpy as np
import tensorflow as tf

# setup data for sequence forecasting (32 samples, 10 days history, 5 features)
X_train = np.random.randn(32, 10, 5)
y_train = np.random.randn(32, 1)  # target prediction

# build and compile the GRU model
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(64, input_shape=(10, 5)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# train and display architecture summary
model.summary()
history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
print(f"Training Loss: {history.history['loss'][0]:.4f}")
