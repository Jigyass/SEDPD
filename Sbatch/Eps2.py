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
import pandas as pd
from scipy.special import comb
from scipy.stats import binom


# In[2]:


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

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Shuffle within batches
attack_loader = DataLoader(attack_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Total training samples: {len(train_dataset)}")
print(f"Training samples after split: {len(train_data)}")
print(f"Attack samples: {len(attack_data)}")
print(f"Testing samples: {len(test_dataset)}")


# In[3]:


def get_pixel_coords(flat_indices, width):
    return [divmod(idx, width) for idx in flat_indices]


# In[4]:


def calculate_pixel_frequencies_from_loader(data_loader, pixel_coords):
    """
    Calculate the frequency of grayscale pixel values at specific coordinates from an MNIST DataLoader.

    Args:
        data_loader (DataLoader): A DataLoader containing the MNIST dataset.
        pixel_coords (list of tuples): A list of (x, y) pixel coordinates to evaluate.

    Returns:
        dict: A dictionary where keys are pixel coordinates, and values are dictionaries of grayscale frequencies.
    """
    pixel_freq = {coord: {} for coord in pixel_coords}

    for batch_idx, (images, _) in enumerate(data_loader):
        # Move batch to CPU and convert to NumPy for efficient processing
        images = images.cpu().numpy()  # Shape: (batch_size, 1, height, width)

        # Iterate through the batch of images
        for img_idx, img_array in enumerate(images):
            img_array = img_array.squeeze(0)  # Convert from (1, H, W) to (H, W)

            # Check and count each specified pixel coordinate
            for (i, j) in pixel_coords:
                if 0 <= i < img_array.shape[0] and 0 <= j < img_array.shape[1]:
                    pixel_value = int(img_array[i, j] * 255)  # Convert to grayscale intensity (0-255)
                    
                    if pixel_value not in pixel_freq[(i, j)]:
                        pixel_freq[(i, j)][pixel_value] = 0
                    pixel_freq[(i, j)][pixel_value] += 1  # Increment count

        print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

    return pixel_freq


# In[5]:


top_22_coords = [
    (151, 83), (155, 83), (147, 79), (87, 79), (151, 79),
    (151, 87), (155, 87), (139, 71), (139, 75), (91, 79),
    (147, 83), (143, 75), (89, 75), (83, 79), (147, 75),
    (108, 146), (115, 156), (167, 158), (85, 75), (155, 79),
    (171, 158), (125, 77)
]

pixel_freq = calculate_pixel_frequencies_from_loader(train_loader, top_22_coords)

print(pixel_freq[(151, 83)])


# In[6]:


import pandas as pd

def aggregate_grayscale_frequencies(pixel_freq):
    """
    Aggregate grayscale frequencies from pixel frequency data.

    Args:
        pixel_freq (dict): Dictionary of pixel frequencies with coordinates as keys
                           and grayscale intensity counts as values.

    Returns:
        pd.DataFrame: DataFrame containing aggregated frequencies for each pixel.
    """
    data = []

    # Convert pixel frequency data into a flat list for DataFrame
    for (i, j), gray_counts in pixel_freq.items():
        for gray_value, count in gray_counts.items():
            data.append((i, j, gray_value, count))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['x', 'y', 'gray_value', 'frequency'])

    # Group by pixel coordinates (x, y) and aggregate frequencies
    result = df.groupby(['x', 'y', 'gray_value'])['frequency'].sum().reset_index()

    return result


# In[7]:


# Assuming `pixel_freq` is the output from `calculate_pixel_frequencies_from_loader`
result_df = aggregate_grayscale_frequencies(pixel_freq)

# Display the result
print(result_df)


# In[8]:


def analyze_max_x_for_epsilon(df, t, epsilon):
    """
    Analyze and compute the maximum x for epsilon for each pixel in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing pixel data with columns 'x', 'y', 'gray_value', 'frequency'.
        t (int): Threshold value for frequency adjustment.
        epsilon (float): Epsilon value to determine maximum x.

    Returns:
        pd.DataFrame: DataFrame containing 'x', 'y', 'gray_value', and 'max_x'.
    """
    def max_x_for_epsilon(freq, t, epsilon):
        """
        Compute max_x based on binomial probability constraint.
        """
        remaining_count = int(freq - t)
        if remaining_count <= 0:
            return 0
        
        for x in range(remaining_count + 1):
            # Compute probability using Binomial CDF
            probability = binom.cdf(x, freq, 0.5)  # Assuming uniform probability of success (p=0.5)
            if probability >= epsilon:
                return x
        return remaining_count  # If no value satisfies the condition, return max possible x

    # Apply function to the DataFrame
    df['max_x'] = df.apply(lambda row: max_x_for_epsilon(row['frequency'], t, epsilon), axis=1)

    return df[['x', 'y', 'gray_value', 'max_x']]


# In[9]:


results_df = analyze_max_x_for_epsilon(result_df, t=2, epsilon=2)
maxValues = results_df.max()
print(maxValues)


# In[ ]:


import random
import pickle
import pandas as pd

def sample_grayscale_values(pixel_freq, pixel_coords, results_df, original_df, save_path=None):
    """
    Sample grayscale values based on max_x for each pixel coordinate and optionally save them.

    Args:
        pixel_freq (dict): Dictionary of pixel frequencies with coordinates as keys
                           and grayscale frequency counts as values.
        pixel_coords (list of tuples): List of pixel coordinates to evaluate.
        results_df (pd.DataFrame): DataFrame containing 'x', 'y', 'gray_value', and 'max_x'.
        original_df (pd.DataFrame): Original DataFrame containing pixel grayscale and frequency data.
        save_path (str, optional): Path to save the sampled grayscale values. If provided, saves the result.

    Returns:
        dict: Dictionary of sampled grayscale values for each coordinate.
    """
    # Initialize dictionary to store sampled grayscale values
    sampled_grayscale_values = {coord: {} for coord in pixel_coords}

    # Iterate through the pixel coordinates
    for (i, j) in pixel_coords:
        # Filter the results DataFrame for the current coordinate
        coord_df = results_df[(results_df['x'] == i) & (results_df['y'] == j)]

        # Iterate through the rows for this coordinate
        for _, row in coord_df.iterrows():
            gray_value, max_x = int(round(row['gray_value'])), int(row['max_x'])

            # Filter the original DataFrame for matching grayscale and pixel coordinates
            original_coord_df = original_df[
                (original_df['x'] == i) & 
                (original_df['y'] == j) & 
                (original_df['gray_value'].round().astype(int) == gray_value)
            ]

            # Extract grayscale values from pixel_freq for the matching coordinate
            grayscale_values = []
            for _, orig_row in original_coord_df.iterrows():
                intensity = int(round(orig_row['gray_value']))
                if intensity in pixel_freq[(i, j)]:
                    grayscale_values.extend([intensity] * pixel_freq[(i, j)][intensity])  # Replicate values by frequency

            # Sample up to max_x grayscale values, ensuring no error if fewer values exist
            sampled_grayscale_values[(i, j)][gray_value] = random.sample(grayscale_values, min(len(grayscale_values), max_x)) if grayscale_values else []

    # Save the sampled grayscale values if a save path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(sampled_grayscale_values, f)
        print(f"Sampled grayscale values saved to {save_path}")

    return sampled_grayscale_values


# In[ ]:


import time

start_time = time.time()

sampled_values = sample_grayscale_values(pixel_freq, top_22_coords, results_df, result_df, save_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Mni/t2e2_mni.pkl")

end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")


# In[ ]:




