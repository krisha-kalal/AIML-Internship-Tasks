import numpy as np

# 1. Inputs (X) and Target Outputs (y)
X = np.array([1.0, 2.0])
y = np.array([2.0, 4.0])

# 2. Initialize Model Parameters (Weights & Bias)
w = 0.0
b = 0.0

# 3. Hyperparameters
learning_rate = 0.1
epochs = 100

print(f"Initial: w = {w:.4f}, b = {b:.4f} | Loss = {np.mean((w * X + b - y)**2):.4f}\n")

# 4. Gradient Descent Loop
for epoch in range(1, epochs + 1):
    # Step A: Forward Pass (Line equation: y_pred = w * X + b)
    y_pred = w * X + b
    
    # Step B: Calculate Mean Squared Error (MSE) Loss
    loss = np.mean((y_pred - y) ** 2)
    
    # Step C: Compute Gradients (Derivatives with respect to w and b)
    # dL/dw = (2/N) * sum((y_pred - y) * X)
    # dL/db = (2/N) * sum(y_pred - y)
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))
    
    # Step D: Update Parameters (Gradient Descent Step)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: w = {w:.4f}, b = {b:.4f} | Loss = {loss:.6f}")

print(f"\nFinal Equation: y = {w:.4f} * x + {b:.4f}")
print(f"Final Loss: {loss:.6f}")