#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/ConvModels/Conv_Imagenette.mod', weights_only=False, map_location="cuda:0")

# Move the model to the appropriate device
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")


# In[7]:


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ConvMixer input size
    transforms.ToTensor()
])

# Load the dataset
dataset = datasets.Imagenette(root='/home/j597s263/scratch/j597s263/Datasets/imagenette', download=False, transform=transform)

# Shuffle indices with a fixed random seed for reproducibility
random.seed(42)  # Use any fixed seed for consistency
indices = list(range(len(dataset)))
random.shuffle(indices)

# Split shuffled indices into training and testing
train_indices = indices[:7568]
test_indices = indices[7568:8522]
attack_indices = indices[8522:]

# Create Subsets
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)
attack_data = Subset(dataset, attack_indices)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Shuffle within batches
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)  # No shuffle for test set
attack_loader = DataLoader(attack_data, batch_size=16, shuffle=True)

# Print dataset sizes
print(f"Attack samples: {len(attack_data)}")
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")


# In[11]:


import torch
from lime.lime_image import LimeImageExplainer
import numpy as np
from torchvision.transforms.functional import normalize, resize

# Ensure the model is in evaluation mode and on the correct device
model.eval()
device = 'cuda'
model = model.to(device)

# Define a function for LIME to use for predictions
def predict_function(images):
    # Convert images to tensors and normalize
    tensors = torch.stack([torch.tensor(image).permute(2, 0, 1) for image in images]).to(device)  
    with torch.no_grad():
        outputs = model(tensors)  # Get logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Initialize the LIME explainer
explainer = LimeImageExplainer()

# File to save all explanations
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/lime_ConvImag.npy" 

# Initialize a list to store explanations
lime_explanations = []

# Process the attack_loader
for idx, (image_tensor, label) in enumerate(attack_loader):
    # Handle the batch dimension properly
    for img_idx in range(image_tensor.size(0)):  # Iterate over batch
        single_image_tensor = image_tensor[img_idx]  # Extract single image tensor
        single_label = label[img_idx].item()  # Extract corresponding label

        # Convert the image tensor to HWC format (required by LIME)
        image = single_image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        # Generate LIME explanation
        explanation = explainer.explain_instance(
            image,                    # Input image (HWC format)
            predict_function,         # Prediction function
            labels=(single_label,),   # Ground truth label to explain
            top_labels=1,             # LIME will include the top predicted label
            hide_color=0,             # Color to hide (optional)
            num_samples=1000          # Number of perturbations
        )

        # Determine the label to explain
        if single_label in explanation.local_exp:
            label_to_explain = single_label  # Ground truth label
        else:
            # Use the top predicted label if the ground truth label is not available
            label_to_explain = list(explanation.local_exp.keys())[0]
            print(f"Ground truth label {single_label} not in explanation. Using top predicted label {label_to_explain}.")

        # Save the explanation mask for the selected label
        _, mask = explanation.get_image_and_mask(
            label_to_explain,
            positive_only=True,
            num_features=10,  # Top 10 superpixels
            hide_rest=False
        )

        # Store the mask and label for this image
        lime_explanations.append({'index': idx, 'label': label_to_explain, 'mask': mask})
        print(f"Processed LIME explanation for image {idx}-{img_idx}")

# Save all explanations to a file
np.save(output_file, lime_explanations)
print(f"All LIME explanations saved to {output_file}")


# In[ ]:




