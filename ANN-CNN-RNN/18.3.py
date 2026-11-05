import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# 1. Create a dummy image dataset (4 images, 4x4 pixels, 1 color channel)
# Shape: (samples, height, width, channels)
X = np.random.rand(4, 4, 4, 1).astype(np.float32)

y = np.array([0, 1, 0, 1]) 

# 2. Build the structural layout of the CNN
model = Sequential([
    # Convolutional layer: Scans the image footprint
    Conv2D(filters=2, kernel_size=(2, 2), activation='relu', input_shape=(4, 4, 1)),
    
    # Conversion layer: Flattens the 2D feature matrix into a 1D vector line
    Flatten(),
    
    # Classification layer: Outputs a final probability
    Dense(units=1, activation='sigmoid')
])

# 3. Compile the network (assign optimizer and loss metric)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(X, y, epochs=100, verbose=0)

# 5. Test predictions on the training images
print("Predicted probabilities (closer to 0 or 1):")
print(model.predict(X).round())
