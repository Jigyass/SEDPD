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
from collections import Counter


# In[ ]:


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


# In[ ]:


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvMNIBase.mod', weights_only=False, map_location="cuda:0")

# Move the model to the appropriate device
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")


# In[ ]:


import torch
from lime.lime_image import LimeImageExplainer
import numpy as np
import torchvision.transforms.functional as TF

# Ensure the model is in evaluation mode and on the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Define a function for LIME to use for predictions
def predict_function(images):
    """
    Function for LIME to make model predictions.
    - Converts LIME images back to 1-channel PyTorch tensors.
    - Feeds them into the model.
    - Returns softmax probabilities.
    """
    tensors = []
    for image in images:
        # Convert from HWC (LIME format) to CHW and normalize
        image = image[:, :, 0]  # Extract first channel from (H, W, 3)
        image = np.expand_dims(image, axis=0)  # Convert (H, W) → (1, H, W)
        image = torch.tensor(image, dtype=torch.float32)  # Ensure it's a PyTorch tensor
        tensors.append(image)

    tensors = torch.stack(tensors).to(device)  # Stack all images into a batch
    with torch.no_grad():
        outputs = model(tensors)  # Get logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Initialize the LIME explainer
explainer = LimeImageExplainer()

# File to save explanations
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/Conv/lime_ConvMNI.npy" 

# Store explanations
lime_explanations = []

# Process the attack_loader
for idx, (image_tensor, _) in enumerate(attack_loader):  # Use `_` for unused labels
    for img_idx in range(image_tensor.size(0)):  # Iterate over batch
        single_image_tensor = image_tensor[img_idx]  # Extract single image tensor

        # Convert MNIST grayscale image to HWC format (LIME expects RGB-like format)
        image = single_image_tensor.squeeze(0).cpu().numpy()  # Remove channel dim -> (H, W)
        image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3) to mimic RGB

        # Get the model's predicted label
        single_image_tensor = single_image_tensor.unsqueeze(0).to(device)  # Add batch dim
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

        # Get explanation for the predicted class
        if predicted_class in explanation.local_exp:
            label_to_explain = predicted_class
        else:
            label_to_explain = list(explanation.local_exp.keys())[0]
            print(f"Predicted class {predicted_class} not in explanation. Using top predicted label {label_to_explain}.")

        # Extract LIME mask
        _, mask = explanation.get_image_and_mask(
            label_to_explain,
            positive_only=True,
            num_features=10,  # Top 10 superpixels
            hide_rest=False
        )

        # Store explanation
        lime_explanations.append({'index': idx, 'label': label_to_explain, 'mask': mask})
        print(f"Processed LIME explanation for image {idx}-{img_idx}")

# Save all explanations to a file
np.save(output_file, lime_explanations)
print(f"All LIME explanations saved to {output_file}")

