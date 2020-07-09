import torch
import torchvision
from torchvision.datasets import MNIST

torch.manual_seed(42)

# Download training dataset
dataset = MNIST(root='data/', download=True)
print(len(dataset)) # 60000 images

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset)) # 10000 test images

print(dataset[0]) # 28x28 image with label

import matplotlib.pyplot as plt

image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)

import torchvision.transforms as transforms

# MNIST dataset (images and labels)
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label) # 1x28x28 tensor (1st dim is the color channel)

# Data Splitting
# - Training Set: use this data to train the model.
# - Validation Set: use to evaluate model while training, adjust hyper-parameters, pick
# 	best version of the model.
# - Test Set: use to compare different models or approaches.

from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)

from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size) # no shuffle since not used to train model, only evaluate

# Modelling

import torch.nn as nn

input_size = 28*28
num_classes = 10

# Logistic regression model
#model = nn.Linear(input_size, num_classes)

# Need to reshape/flatten input vectors
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()

print(model.linear.weight.shape, model.linear.bias.shape)
print(list(model.parameters()))

for images, labels in train_loader:
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)

import torch.nn.functional as F

# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)

# Evaluation Metric and Loss Function

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

accuracy(outputs, labels)

# - Accuracy function is not differentiable => cannot use GD to optimize.
# - Use Cross-Entropy instead, defined as: D(yhat,y)=-sum_j(y_j ln(yhat_j))
# - The y and yhat vectors are one-hot encoded.
# - If yhat_j is close to 1, then the negative log is close to 0 and positive.
# - If yhat_j is close to 0, then the negative lof is large and positive. 
# - Multiplying y_j and ln(yhat_j) element by element allows to select only the predicted
#  probability for the correct label.
# - If the true label has a very low predicted probability, then the loss will be large.
# - Take the average over all outputs j to get the total loss per batch.
# - It is now possible to use GD (Cross-Entropy function is differentiable) to minimize
# the loss by tuning the weights to obtain the highest predicted probabilities
# for the correct classes.

loss_fn = F.cross_entropy

# Loss for current batch of data
loss = loss_fn(outputs, labels)
print(loss) 

# To interpret loss: e.g. loss = 2.23 -> exp(-2.23) = 0.1 is
# the predicted probability of the correct label on average.

# Training the Model
# We add a validation phase at each epoch to evaluate the model at each epoch
# Loss fct and metric can be adapted to a specific problem

# General Flow:
#for epoch in range(num_epochs):
#	 Training phase
#    for batch in train_loader:
        # Generate predictions
        # Calculate loss
        # Compute gradients
        # Update weights
        # Reset gradients

    # Validation phase
 #   for batch in val_loader:
        # Generate predictions
        # Calculate loss
        # Calculate metrics (accuracy etc.)
    # Calculate average validation loss & metrics

    # Log epoch, loss & metrics for inspection


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Initial Accuracy
result0 = evaluate(model, val_loader)
print(result0)

# Train for 5 epochs
history1 = fit(5, 0.001, model, train_loader, val_loader) # 0.79 accuracy
# Train for 5 more epochs
history2 = fit(5, 0.001, model, train_loader, val_loader) # 0.82 accuracy
# Train for 5 more epochs
history3 = fit(5, 0.001, model, train_loader, val_loader) # 0.84 accuracy
# Train for 5 more epochs
history4 = fit(5, 0.001, model, train_loader, val_loader) # 0.85 accuracy

# Changing the learning rate (or batch size) and increase nb of training
# epochs could improve accuracy a bit.
# The linear model assumption is probably unable to capture non-linear
# relationships in the data and thus accuracy is limited.

# Accuracy vs epoch plot
history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

# Checking individual images
# Define test dataset
test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor())

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label)

img.unsqueeze(0).shape # adds a dimension: 1x28x28 -> 1x1x28x28 so that model views
# image as a batch containing one image.

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

# Test 1
img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# Test 2
img, label = test_dataset[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# Test 3
img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# Test 4
img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

# Model Test Loss and Accuracy
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
result

# Saving and Loading Model Parameters
torch.save(model.state_dict(), 'mnist-logistic.pth')

model.state_dict()

model2 = MnistModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()

# Sanity check
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
result


