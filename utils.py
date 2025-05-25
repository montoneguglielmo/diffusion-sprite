import torch
import os
from torchvision.utils import make_grid, save_image
import numpy as np

def save_intermediate_grids(intermediate_images, output_dir="intermediate_grids"):
    """
    Create and save grid visualizations of intermediate images for each timestep.
    
    Args:
        intermediate_images (numpy.ndarray): Array of shape (T, N, 3, 16, 16) containing intermediate images
            T: number of timesteps
            N: number of samples
            3: number of channels (RGB)
            16, 16: image dimensions
        output_dir (str): Directory where to save the grid images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy array to torch tensor
    if isinstance(intermediate_images, np.ndarray):
        intermediate_images = torch.from_numpy(intermediate_images).float()
    
    # For each timestep
    for t in range(intermediate_images.shape[0]):
        # Get all samples for this timestep
        timestep_images = intermediate_images[t]  # Shape: (N, 3, 16, 16)
        
        # Create a grid of images
        grid = make_grid(timestep_images, nrow=int(timestep_images.shape[0] ** 0.5), padding=2)
        
        # Save the grid
        save_image(grid, os.path.join(output_dir, f"grid_timestep_{t:03d}.png"))
