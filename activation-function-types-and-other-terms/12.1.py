import numpy as np

# 1. Secret rule: y = 2 * X (Model doesn't know the '2' yet)
X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])

# 2. Settings (Hyperparameters)
learning_rate = 0.1
weight = 0.0  # Our starting guess for the multiplier

# 3. The Training Loop (Epochs)
for epoch in range(1, 6):  # Run 5 passes over the data
    # Step A: Guess the answer (Linear Activation)
    y_pred = weight * X

    # Step B: Calculate the average mistake (Loss)
    loss = np.mean((y - y_pred) ** 2)

    # Step C: Figure out the direction to change the guess
    direction = -2 * np.mean(X * (y - y_pred))

    # Step D: Update the guess using the learning rate
    weight -= learning_rate * direction

    # Print the progress of each epoch
    print(f"Epoch {epoch} | Loss: {loss:.4f} | Current Weight Guess: {weight:.2f}")