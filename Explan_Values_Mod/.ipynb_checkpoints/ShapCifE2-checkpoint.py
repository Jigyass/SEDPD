#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

dataset_path = "/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvCifE2.pt"

modified_dataset = torch.load(dataset_path, map_location="cuda", weights_only=False)

images = modified_dataset["images"]  
labels = modified_dataset["labels"]  

modified_dataset = TensorDataset(images, labels)
modified_loader = DataLoader(modified_dataset, batch_size=128, shuffle=False)

model_path = "/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvCifar.mod"
model = torch.load(model_path, weights_only=False)
model = model.to('cuda')
model.eval()

print("Model and dataset loaded successfully!")

device = 'cuda'
baseline_images, _ = next(iter(modified_loader))
baseline_images = baseline_images[:50].to(device)  

explainer = shap.GradientExplainer(model, baseline_images)

shap_values = np.zeros((1, 224, 224, 10))  

for batch_images, _ in modified_loader:
    batch_images = batch_images.to(device)

    batch_shap_values = explainer.shap_values(batch_images)  # Shape: (batch_size, C, H, W, num_classes)

    for i in range(len(batch_images)):
        shap_values += batch_shap_values[i]

    print(f"{len(batch_images)} images processed in this batch")

print("SHAP value aggregation completed!")

# Save SHAP values
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/ConvCifDef/DTE2.npy"
np.save(output_file, shap_values)
print(f"Aggregated SHAP values saved to {output_file}")

