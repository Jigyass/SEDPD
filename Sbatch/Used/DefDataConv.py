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


# In[14]:


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

# Create Subsets
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Shuffle within batches
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)  # No shuffle for test set

# Print dataset sizes
print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

data_loader = train_loader


# In[3]:


def get_pixel_coords(flat_indices, width):
    return [divmod(idx, width) for idx in flat_indices]


# In[4]:


def calculate_pixel_frequencies_from_loader(data_loader, pixel_coords):
    """
    Calculate the frequency of pixel values at specific coordinates from a DataLoader.

    Args:
        data_loader (DataLoader): A DataLoader containing the image dataset.
        pixel_coords (list of tuples): A list of (x, y) pixel coordinates to evaluate.

    Returns:
        dict: A dictionary where keys are pixel coordinates, and values are dictionaries of RGB frequencies.
    """
    pixel_freq = {coord: {} for coord in pixel_coords}

    for batch_idx, (images, _) in enumerate(data_loader):
        # Move batch to CPU for processing if it's on GPU
        images = images.cpu()

        # Iterate through the batch of images
        for img_idx, img_tensor in enumerate(images):
            # Convert to numpy array for easy access
            img_array = img_tensor.permute(1, 2, 0).numpy()  # (height, width, 3)

            # Check and count each specified pixel coordinate
            for (i, j) in pixel_coords:
                if i < img_array.shape[0] and j < img_array.shape[1]:
                    pixel = tuple(img_array[i, j])  # Extract RGB tuple
                    if pixel not in pixel_freq[(i, j)]:
                        pixel_freq[(i, j)][pixel] = []
                    pixel_freq[(i, j)][pixel].append((batch_idx * len(images) + img_idx, pixel))

        print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

    return pixel_freq


# In[5]:


# Define the top 22 coordinates
top_22_coords = [
    (118, 178), (100, 181), (75, 164), (137, 103), (126, 78),
    (74, 175), (146, 110), (86, 46), (158, 98), (90, 173),
    (106, 134), (84, 165), (97, 45), (74, 174), (77, 163),
    (84, 110), (90, 174), (137, 87), (86, 106), (186, 142),
    (74, 173), (138, 87)
]

# Calculate pixel frequencies
pixel_freq = calculate_pixel_frequencies_from_loader(train_loader, top_22_coords)

# Inspect results for a specific coordinate
print(pixel_freq[(118, 178)])


# In[6]:


def aggregate_rgb_frequencies(pixel_freq):
    """
    Aggregate RGB frequencies from pixel frequency data and convert RGB to grayscale.

    Args:
        pixel_freq (dict): Dictionary of pixel frequencies with coordinates as keys
                           and RGB frequency counts as values.

    Returns:
        pd.DataFrame: DataFrame containing aggregated frequencies for each pixel and
                      grayscale mapping.
    """
    data = []

    # Convert pixel frequency data into a flat list for DataFrame
    for (i, j), rgb_counts in pixel_freq.items():
        for rgb, count in rgb_counts.items():
            data.append((i, j, rgb[0], rgb[1], rgb[2], len(count)))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['x', 'y', 'R', 'G', 'B', 'frequency'])

    # Convert RGB to grayscale
    def rgb_to_grayscale(r, g, b):
        return 0.299 * r + 0.587 * g + 0.114 * b

    df['gray_value'] = df.apply(lambda row: rgb_to_grayscale(row['R'], row['G'], row['B']), axis=1)

    # Group by pixel coordinates (x, y)
    grouped = df.groupby(['x', 'y'])

    # Aggregate grayscale and frequency information
    result = grouped.apply(
        lambda group: group.groupby('gray_value')
        .agg({'frequency': 'sum', 'R': 'first', 'G': 'first', 'B': 'first'})
        .reset_index()
        .assign(x=group['x'].iloc[0], y=group['y'].iloc[0])
    ).reset_index(drop=True)

    return result


# In[7]:


# Assuming `pixel_freq` is the output from `calculate_pixel_frequencies_from_loader`
result_df = aggregate_rgb_frequencies(pixel_freq)

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
        # Remaining count after subtracting the threshold t
        remaining_count = int(freq - t)
        if remaining_count <= 0:
            return 0

        max_x = 0
        for x in range(1, remaining_count + 1):
            # Compute the probability using scipy's comb
            probability = comb(freq, x) / comb(remaining_count, x)
            if probability <= epsilon:
                max_x = x
            else:
                break
        return max_x

    # Process each row in the DataFrame and calculate max_x
    results = []
    for _, row in df.iterrows():
        max_x = max_x_for_epsilon(row['frequency'], t, epsilon)
        results.append((row['x'], row['y'], row['gray_value'], max_x))

    # Create a new DataFrame with the results
    return pd.DataFrame(results, columns=['x', 'y', 'gray_value', 'max_x'])


# In[9]:


results_df = analyze_max_x_for_epsilon(result_df, t=2, epsilon=2)
maxValues = results_df.max()
print(maxValues)


# In[10]:


import random
import pickle

def sample_rgb_values(pixel_freq, pixel_coords, results_df, original_df, save_path=None):
    """
    Sample RGB values based on max_x for each pixel coordinate and optionally save them.

    Args:
        pixel_freq (dict): Dictionary of pixel frequencies with coordinates as keys
                           and RGB frequency counts as values.
        pixel_coords (list of tuples): List of pixel coordinates to evaluate.
        results_df (pd.DataFrame): DataFrame containing 'x', 'y', 'gray_value', and 'max_x'.
        original_df (pd.DataFrame): Original DataFrame containing pixel RGB and frequency data.
        save_path (str, optional): Path to save the sampled RGB values. If provided, saves the result.

    Returns:
        dict: Dictionary of sampled RGB values for each coordinate and grayscale level.
    """
    # Initialize a dictionary to store sampled RGB values
    sampled_rgb_values = {coord: {} for coord in pixel_coords}

    # Iterate through the pixel coordinates
    for (i, j) in pixel_coords:
        # Filter the results DataFrame for the current coordinate
        coord_df = results_df[(results_df['x'] == i) & (results_df['y'] == j)]

        # Iterate through the rows for this coordinate
        for _, row in coord_df.iterrows():
            gray_value, max_x = row['gray_value'], row['max_x']

            # Filter the original DataFrame for matching grayscale and pixel coordinates
            original_coord_df = original_df[
                (original_df['x'] == i) & 
                (original_df['y'] == j) & 
                (original_df['gray_value'].round().astype(int) == int(round(gray_value)))
            ]

            # Extract RGB values from pixel_freq for the matching grayscale and coordinate
            rgb_values = []
            for _, orig_row in original_coord_df.iterrows():
                rgb_key = (orig_row['R'], orig_row['G'], orig_row['B'])
                if rgb_key in pixel_freq[(i, j)]:
                    rgb_values.extend([entry[0] for entry in pixel_freq[(i, j)][rgb_key]])

            # Sample up to max_x RGB values, or return an empty list if insufficient values
            sampled_rgb_values[(i, j)][gray_value] = random.sample(rgb_values, int(max_x)) if len(rgb_values) >= max_x else []

    # Save the sampled RGB values if a save path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(sampled_rgb_values, f)
        print(f"Sampled RGB values saved to {save_path}")

    return sampled_rgb_values


# In[11]:


'''import time

start_time = time.time()

sampled_values = sample_rgb_values(pixel_freq, top_22_coords, results_df, result_df, save_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Sampled_Values/t2e2.pkl")

end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")'''


# In[12]:


import os
import torch

def apply_samples_to_dataset(data_loader, sampled_rgb_values, pixel_coords, output_path):
    """
    Apply sampled RGB values to images and save the dataset with labels to a file.

    Args:
        data_loader (DataLoader): DataLoader containing the images to modify.
        sampled_rgb_values (dict): Dictionary of sampled RGB values for each pixel.
        pixel_coords (list of tuples): List of pixel coordinates to evaluate.
        output_path (str): Path to the file where the dataset will be saved.
    """
    modified_images = []
    labels = []

    # Process each image in the data loader
    for batch_idx, (images, batch_labels) in enumerate(data_loader):
        images = images.clone()  # Clone to avoid modifying the original data
        batch_size = images.size(0)

        for img_idx in range(batch_size):
            image_tensor = images[img_idx]
            img_array = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)

            height, width, _ = img_array.shape
            for coord in pixel_coords:
                x, y = coord
                if x < height and y < width:
                    if (x, y) in sampled_rgb_values:
                        gray_levels = sampled_rgb_values[(x, y)]
                        found = False
                        for gray_level in gray_levels:
                            if img_idx in gray_levels[gray_level]:
                                found = True
                                break
                        if not found:
                            img_array[x, y] = [0, 0, 0]  # Set to black if no match
                    else:
                        img_array[x, y] = [0, 0, 0]  # Set to black for unmatched coordinates

            # Convert modified array back to tensor
            modified_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

            # Add modified tensor and corresponding label to the dataset
            modified_images.append(modified_tensor)
            labels.append(batch_labels[img_idx].item())

        print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

    # Save the modified dataset
    dataset = {
        "images": torch.stack(modified_images),
        "labels": torch.tensor(labels)
    }
    torch.save(dataset, output_path)
    print(f"Modified dataset saved to {output_path}")


# In[15]:


'''output_dir = "/home/j597s263/scratch/j597s263/Datasets/Defense/DefenseCM.pt"
apply_samples_to_dataset(train_loader, sampled_values, top_22_coords, output_dir)'''


# In[16]:


import pickle

pixel_coords = top_22_coords

# Function calls for each dataset
sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e2.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE2.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e3.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE3.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e4.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE4.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e5.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE5.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e6.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE6.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e7.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE7.pt"
)

sampled_rgb_values = pickle.load(open("/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/Sampled_Values/Img/t2e8.pkl", "rb"))
apply_samples_to_dataset(
    data_loader=data_loader,
    sampled_rgb_values=sampled_rgb_values,
    pixel_coords=pixel_coords,
    output_path="/home/j597s263/scratch/j597s263/Datasets/Defense/Conv/ConvImgE8.pt"
)

