import torch
import numpy as np

def model(x):
	return x @ w.t() + b

def mse(yhat, y):
	return torch.mean((yhat-y)**2)

'''
# Alternative to manually defining mse
import torch.nn as nn
mse = nn.MSELoss()
'''

# Example data
x = inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32') # Input (temp, rainfall, humidity)
y = targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32') # Targets (apples, oranges)

x = torch.from_numpy(x)
y = torch.from_numpy(y)

# Initialize weights and biases
torch.manual_seed(42)
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


# Train for 100 epochs
lr = 1e-5

for epoch in range(500):
    yhat = model(x)
    loss = mse(yhat, y)
    loss.backward() # computes gradients wrt to weights and biases (requires_grad=True)
    with torch.no_grad(): # no tracking, calculating or modifying gradients while updating the weights and biases.
        w -= w.grad * lr
        b -= b.grad * lr
        w.grad.zero_() # reset gradients to 0 since pytorch accumulates gradients
        b.grad.zero_()

yhat = model(x)
loss = mse(yhat, y)
print(loss)
print(y)
print(yhat)
