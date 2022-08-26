# Linear regression example to predict crop yield of apples and oranges
# by looking at the temp., humidity, and amount of rainfall in that region

# yield_apple = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2

import numpy as np
import torch

# Inputs (Temp., Rainfall, Humidity)
inputs = np.array([
    [73, 67, 43],
    [91, 88, 64],
    [87, 134, 58],
    [102, 43, 37],
    [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([
    [56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119]], dtype='float32')

# Convert inputs and targets to Tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print()
print(targets)

# ==== Weights and biases ====
# w ==> weights
# b ==> biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


# @ ==> matrix multiplication
# w.t() ==> transpose of matrix `w`
def model(x):
    return x @ w.t() + b



# Now to calculate difference by making a loss function:
# MSE loss -> mean squared loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# ====== Gradient descent method =======
# Train for 100 epochs to lower the loss
for i in range(350):
    # New loss
    preds = model(inputs)
    loss = mse(preds, targets)
    # Compute gradients
    loss.backward()

    # First reset both weight and bias gradients as zero
    with torch.no_grad():
        w -= w.grad * 1e-4
        b -= b.grad * 1e-4
        w.grad.zero_()
        b.grad.zero_()

# Now verify the value of loss
preds = model(inputs)
loss = mse(preds, targets)
print("\nLoss:\t", loss)
print("\nPredictions:\t", preds)
print("\nTargets:\t", targets)
