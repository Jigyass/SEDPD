#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader, TensorDataset


# Residual Block
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# ConvMixer Model
def ConvMixer():
    dim = 256         
    depth = 8         
    kernel_size = 5    
    patch_size = 4     
    n_classes = 10    

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


dataset_path = "/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvCifE8.pt"
modified_dataset = torch.load(dataset_path, map_location="cuda", weights_only=False)

images = modified_dataset["images"]  
labels = modified_dataset["labels"]  

modified_dataset = TensorDataset(images, labels)
modified_loader = DataLoader(modified_dataset, batch_size=128, shuffle=False)

print("Defense dataset loaded successfully!")


model_path = "/home/j597s263/scratch/j597s263/Models/ConvModels/Base/ConvCifar.mod"

# Ensure model architecture is defined
model = torch.load(model_path, map_location="cuda", weights_only=False)
model = model.to('cuda')
model.eval()

print("Model loaded successfully!")


device = 'cuda' 
model.to(device)
model.eval()

# Define function for LIME predictions
def predict_function(images):
    """
    Converts images to tensors, normalizes, and returns softmax probabilities.
    """
    tensors = []
    for image in images:
        # Convert from HWC (LIME format) to CHW
        image = np.moveaxis(image, -1, 0)  # Change (H, W, 3) → (3, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        tensors.append(image)

    tensors = torch.cat(tensors).to(device)  
    with torch.no_grad():
        outputs = model(tensors)  
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Initialize LIME explainer
explainer = LimeImageExplainer()
lime_file_path = "/home/j597s263/scratch/j597s263/Datasets/Explanation_values/ConvCifDef/DTE8_Lime.npy"

lime_explanations = []

for idx, (image_tensor, _) in enumerate(modified_loader):  
    for img_idx in range(image_tensor.size(0)):  
        single_image_tensor = image_tensor[img_idx]  

        # Convert (3, 224, 224) → (224, 224, 3) for LIME
        image = single_image_tensor.permute(1, 2, 0).cpu().numpy()

        # Get model's predicted label
        single_image_tensor = single_image_tensor.unsqueeze(0).to(device)  
        outputs = model(single_image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

        # Generate LIME explanation
        explanation = explainer.explain_instance(
            image,
            predict_function,
            labels=(predicted_class,),
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        # Get explanation mask
        _, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=10,  
            hide_rest=False
        )

        # Store LIME explanation
        lime_explanations.append({'index': idx, 'label': predicted_class, 'mask': mask})
        print(f"Processed LIME explanation for image {idx}-{img_idx}")

# Save LIME explanations
np.save(lime_file_path, lime_explanations)
print(f"LIME explanations saved to {lime_file_path}")

