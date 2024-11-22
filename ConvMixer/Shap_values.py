import os
import random
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
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

def main():
    # Load the model
    model_path = '/home/j597s263/scratch/j597s263/Models/Conv_Imagenette.mod'
    model = torch.load(model_path, weights_only=False, map_location="cuda:0")
    model = model.to('cuda')
    model.eval()
    print("Model loaded successfully!")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ConvMixer input size
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = datasets.Imagenette(root='/home/j597s263/scratch/j597s263/Datasets/imagenette', 
                                   download=False, transform=transform)

    # Shuffle indices with a fixed random seed for reproducibility
    random.seed(42)  # Use any fixed seed for consistency
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Split shuffled indices into training, testing, and attack datasets
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
    attack_loader = DataLoader(attack_data, batch_size=8, shuffle=True)

    # Print dataset sizes
    print(f"Attack samples: {len(attack_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    device = 'cuda'

    # Use a subset from the training data as the baseline
    baseline_images, _ = next(iter(train_loader))  # Take a batch from train_loader
    baseline_images = baseline_images[:50]  # Limit to 50 images for efficiency
    baseline_images = baseline_images.to(device)

    # Initialize SHAP Gradient Explainer
    explainer = shap.GradientExplainer(model, baseline_images)

    # Initialize aggregation array for SHAP values
    shap_values = np.zeros((3, 224, 224, 10))  # (Channels, Height, Width, Classes)

    # Process attack images in batches
    for batch_images, _ in attack_loader:
        batch_images = batch_images.to(device)

        # Compute SHAP values for the batch
        batch_shap_values = explainer.shap_values(batch_images)  # Shape: (batch_size, 3, 224, 224, 10)

        # Sum the SHAP values into the aggregation array
        for i in range(len(batch_images)):
            shap_values += batch_shap_values[i]

        print(f"{len(batch_images)} images processed in this batch")

    # Normalize SHAP values by the number of attack images
    shap_values /= len(attack_loader.dataset)

    print("SHAP value aggregation completed!")

    # Save SHAP values to a file
    output_file = '/home/j597s263/scratch/j597s263/Results/aggregated_shap_values.npy'
    np.save(output_file, shap_values)
    print(f"Aggregated SHAP values saved to {output_file}")

if __name__ == "__main__":
    main()
