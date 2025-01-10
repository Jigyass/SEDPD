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


# In[7]:


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvCifar.mod', weights_only=False, map_location="cuda:0")

# Move the model to the appropriate device
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")


# In[2]:


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
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # Shuffle within batches
attack_loader = DataLoader(attack_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Print dataset sizes
print(f"Original training samples: {len(train_dataset)}")
print(f"Training samples after split: {len(train_data)}")
print(f"Validation samples: {len(attack_data)}")
print(f"Testing samples (unchanged): {len(test_dataset)}")


# In[10]:


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
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/Conv/lime_ConvCif.npy" 

# Initialize a list to store explanations
lime_explanations = []

# Process the attack_loader
for idx, (image_tensor, _) in enumerate(attack_loader):  # Use _ for unused label
    # Handle the batch dimension properly
    for img_idx in range(image_tensor.size(0)):  # Iterate over batch
        single_image_tensor = image_tensor[img_idx]  # Extract single image tensor

        # Convert the image tensor to HWC format (required by LIME)
        image = single_image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

        # Get the model's predicted label
        single_image_tensor = single_image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(single_image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

        # Generate LIME explanation
        explanation = explainer.explain_instance(
            image,                    # Input image (HWC format)
            predict_function,         # Prediction function
            labels=(predicted_class,),# Predicted label to explain
            top_labels=1,             # LIME will include the top predicted label
            hide_color=0,             # Color to hide (optional)
            num_samples=1000          # Number of perturbations
        )

        # Check if the predicted class is in the explanation
        if predicted_class in explanation.local_exp:
            label_to_explain = predicted_class  # Use the predicted class
        else:
            # Use the top predicted label if the predicted class is not available
            label_to_explain = list(explanation.local_exp.keys())[0]
            print(f"Predicted class {predicted_class} not in explanation. Using top predicted label {label_to_explain}.")

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




