#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast


# In[2]:


# Load CIFAR-10 dataset
train_data = datasets.CIFAR10(root='/home/j597s263/scratch/j597s263/Datasets/cifar10', download=False, transform=ToTensor(), train=True)
test_data = datasets.CIFAR10(root='/home/j597s263/scratch/j597s263/Datasets/cifar10', download=False, transform=ToTensor(), train=False)

# Create DataLoaders for training and testing
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

print(f"Training data: {len(train_data)} samples")
print(f"Testing data: {len(test_data)} samples")
  # Resize to ConvMixer input size


# In[3]:


import random
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

# Load CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root='/home/j597s263/scratch/j597s263/Datasets/cifar10', 
                                 download=False, 
                                 transform=transform, 
                                 train=True)

test_dataset = datasets.CIFAR10(root='/home/j597s263/scratch/j597s263/Datasets/cifar10', 
                                download=False, 
                                transform=transform, 
                                train=False)

random.seed(42)  
train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)

split_idx = int(0.9 * len(train_indices))  
train_indices, attack_indices = train_indices[:split_idx], train_indices[split_idx:]

# Create Subsets
train_data = Subset(train_dataset, train_indices)
attack_data = Subset(train_dataset, attack_indices)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)  # Shuffle within batches
attack_loader = DataLoader(attack_data, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Print dataset sizes
print(f"Original training samples: {len(train_dataset)}")
print(f"Training samples after split: {len(train_data)}")
print(f"Validation samples: {len(attack_data)}")
print(f"Testing samples (unchanged): {len(test_dataset)}")


# In[4]:


import torch.nn as nn

# Residual block
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# ConvMixer model with hard-coded parameters
def ConvMixer():
    dim = 256          # Embedding dimension
    depth = 8          # Number of ConvMixer blocks
    kernel_size = 5    # Kernel size for depthwise convolution
    patch_size = 4     # Patch size for initial convolution
    n_classes = 10     # CIFAR-10 has 10 classes

    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


# In[5]:


model = ConvMixer().to('cuda')


# In[6]:


# Hyperparameters
epochs = 1#70
learning_rate = 0.005
opt_eps = 1e-3
clip_grad = 1.0
device = 'cuda' 

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=opt_eps)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Automatic Mixed Precision (AMP)
scaler = GradScaler()

# Training and Testing Loop
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)

        # Forward and backward pass with AMP
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

    # Log training loss for the epoch
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

    # Testing phase after each epoch
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Log test accuracy and loss
    test_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")


# In[ ]:


torch.save(model, '/home/j597s263/Models/Conv_Cifar.mod')


# In[ ]:




