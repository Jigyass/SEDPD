#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import shap


# In[2]:


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
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
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


# In[3]:


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvMNIBase.mod', weights_only=False, map_location="cuda:0")

# Move the model to the appropriate device
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")


# In[4]:


import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

# Define dataset root directory
mnist_root = '/home/j597s263/scratch/j597s263/Datasets/MNIST'

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root=mnist_root, transform=transform, train=True, download=False)
test_dataset = datasets.MNIST(root=mnist_root, transform=transform, train=False, download=False)

train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)  

split_idx = int(0.9 * len(train_indices))  
train_indices, attack_indices = train_indices[:split_idx], train_indices[split_idx:]

train_data = Subset(train_dataset, train_indices)
attack_data = Subset(train_dataset, attack_indices)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)  # Shuffle within batches
attack_loader = DataLoader(attack_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Total training samples: {len(train_dataset)}")
print(f"Training samples after split: {len(train_data)}")
print(f"Attack samples: {len(attack_data)}")
print(f"Testing samples: {len(test_dataset)}")


# In[ ]:


import torch
import numpy as np
import shap

device = 'cuda' 
model.to(device)
model.eval()

baseline_images, _ = next(iter(train_loader))  
baseline_images = baseline_images[:100] 
baseline_images = baseline_images.to(device)

explainer = shap.GradientExplainer(model, baseline_images)

shap_values = np.zeros((1, 224, 224, 10))  

for batch_images, _ in attack_loader:
    batch_images = batch_images.to(device)

    batch_shap_values = explainer.shap_values(batch_images)  

    for i in range(len(batch_images)):
        shap_values += batch_shap_values[i]

    print(f"{len(batch_images)} images processed in this batch")

print("SHAP value aggregation completed!")

# Save SHAP values to a file
output_file = '/home/j597s263/scratch/j597s263/Datasets/Explanation_values/Conv/SHAP_ConvMNI.npy'
np.save(output_file, shap_values)
print(f"Aggregated SHAP values saved to {output_file}")

