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


import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)


# In[4]:


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/Resnet/Base/ResImageBase.mod', weights_only=False, map_location="cuda")

# Move the model to the appropriate device
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")


# In[5]:


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
attack_loader = DataLoader(attack_data, batch_size=5, shuffle=True)

# Print dataset sizes
print(f"Attack samples: {len(attack_data)}")
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")


# In[8]:


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
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/Resnet/LimeImg.npy" 

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

