#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.amp import GradScaler, autocast
import os
import random
from torch.utils.data import DataLoader, SubsetRandomSampler


# In[ ]:


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_dir = '/home/j597s263/scratch/j597s263/Datasets/TinyImage/tiny-imagenet-200/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)

test_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)

num_train = len(train_dataset)
num_classes = len(train_dataset.classes)  

class_to_indices = {i: [] for i in range(num_classes)}
for idx, (_, label) in enumerate(train_dataset.samples):
    class_to_indices[label].append(idx)

attack_indices = []
num_per_class = (num_train // 10) // num_classes  
for class_idx, indices in class_to_indices.items():
    np.random.shuffle(indices) 
    attack_indices.extend(indices[:num_per_class])

attack_size = len(attack_indices)
remaining_indices = list(set(range(num_train)) - set(attack_indices))  

train_sampler = SubsetRandomSampler(remaining_indices)
attack_sampler = SubsetRandomSampler(attack_indices)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
attack_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Print dataset sizes
print(f"Number of training samples: {len(remaining_indices)}")
print(f"Number of attack samples: {len(attack_indices)}")
print(f"Number of test samples: {len(test_dataset)}")


# In[ ]:


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
    n_classes = 200    # CIFAR-10 has 10 classes

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


# In[ ]:


model = ConvMixer().to('cuda')


# In[ ]:


# Hyperparameters
epochs = 150
learning_rate = 3e-4
opt_eps = 1e-3
clip_grad = 1.0
device = 'cuda'  

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=opt_eps)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate*10,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100,
    steps_per_epoch=30,
    epochs=epochs
)

criterion = nn.CrossEntropyLoss()

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
        with autocast(device_type='cuda'):
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


torch.save(model, '/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvTiny.mod')

