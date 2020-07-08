import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

x = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

y = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

x = torch.from_numpy(x)
y = torch.from_numpy(y)

from torch.utils.data import TensorDataset

# TensorDataset allows to access a small portion of the training data with indexing as a tuple (x, y)
trn = TensorDataset(x, y)

from torch.utils.data import DataLoader

# DataLoader splits the data into batches of predefined size and allows to shuffle/sample the data
batch_size = 5
trn_dl = DataLoader(trn, batch_size, shuffle=True)

# Define model with nn.Linear (initializes weights and biases automatically)
model = nn.Linear(3, 2)

'''
# Parameters
print(model.weight)
print(model.bias)
list(model.parameters())
'''

# Package contains loss functions
import torch.nn.functional as F

# Define loss function
loss_fn = F.mse_loss

# Define optimizer (SGD)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Function to train the model
def fit(num_epochs, model, loss_fn, opt, trn_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in trn_dl:
            
            # 1. Generate predictions
            yhat = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(yhat, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(500, model, loss_fn, opt, trn_dl)

yhat = model(x)
print(yhat)
print(y)




