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

train_dataset = datasets.MNIST(root=mnist_root, transform=transform, train=True, download=True)
test_dataset = datasets.MNIST(root=mnist_root, transform=transform, train=False, download=True)

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

        
        
def ResNet50(num_classes, channels=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)


# In[ ]:


import torch
# Load the entire model
model = torch.load('/home/j597s263/scratch/j597s263/Models/Resnet/Base/ResMNIBase.mod', weights_only=False, map_location="cuda:0")

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
output_file = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/Resnet/LimeMNI.npy" 

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

