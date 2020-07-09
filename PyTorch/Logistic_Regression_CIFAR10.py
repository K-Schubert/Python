'''

- Data: CIFAR-10 dataset contains 60000 32x32 colour images with 10 classes, 6000 images per class.
  There are 50000 training images and 10000 test images. The training batches are of size 10000 (5).
  The test batch contains 1000 random images from each class. The training batches are not necessarily
  balanced.
- Labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Inputs: 32x32 RGB images.
- Remarks: As a first attempt, the RGB images are converted to grayscale to simplify computations during
  training.
- Goal: Train a multinomial logistic regression model to accurately predict the label of an image.
- Results: Accuracy is about 27% after 20 epochs. Increasing training to 100 epochs increases accuracy to .
  The low performance is certainly due to the non-linear relationships in the data (while we assume linearity
  between inputs and the labels). It is obvious a neural network will perform better for this task.

'''


import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

torch.manual_seed(42)

# Download training/test dataset
# CIFAR10 dataset (images and labels) -> CONVERTED TO GRAYSCALE
trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

dataset = CIFAR10(root='data/', 
                train=True,
                transform=trans)

test_dataset = CIFAR10(root='data/', train=False)

'''
import matplotlib.pyplot as plt

image, label = dataset[0]
plt.imshow(image[0,:,:], cmap='gray')
plt.show()
print('Label:', label)
'''

'''
# Check a sample image's dimensions
img_tensor, label = dataset[0]
print(img_tensor.shape, label) # 1x32x32 tensor (1st dim is the color channel -> grayscale -> originally 3x32x32)
'''

from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [40000, 10000])
#len(train_ds), len(val_ds)

from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size) # no shuffle since not used to train model, only evaluate

import torch.nn as nn

input_size = 32*32
num_classes = 10


'''
print(model.linear.weight.shape, model.linear.bias.shape)
print(list(model.parameters()))
'''

import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 32*32)
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
    
model = CIFARModel()

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
history1 = fit(5, 0.001, model, train_loader, val_loader) # 0.24 accuracy
# Train for 5 more epochs
history2 = fit(5, 0.001, model, train_loader, val_loader) # 0.24 accuracy
# Train for 5 more epochs
history3 = fit(5, 0.001, model, train_loader, val_loader) # 0.26 accuracy
# Train for 5 more epochs
history4 = fit(5, 0.001, model, train_loader, val_loader) # 0.27 accuracy
# Train for 80 more epochs
history5 = fit(80, 0.001, model, train_loader, val_loader) #  accuracy

# Accuracy vs epoch plot
history = [result0] + history1 + history2 + history3 + history4 + history5
accuracies = [result['val_acc'] for result in history]

import matplotlib.pyplot as plt

plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');
plt.show()

# Model Test Loss and Accuracy
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
print(result)

# Saving Model Parameters
torch.save(model.state_dict(), 'cifar-logistic.pth')
