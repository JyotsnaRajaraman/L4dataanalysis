import numpy as np
import h5py
import os
from pathlib import Path
import time

def load_h5_with_bbox(bbox, base_dir='.', dataset_name='data'):
    """
    Load data from H5 files based on a specified bounding box.
    
    Files are assumed to follow the naming convention 'x{x_idx}y{y_idx}z{z_idx}.h5',
    where each file contains a 1024x1024x1024 block of data, and indexing starts at 0.
    
    Args:
        bbox (tuple): A tuple of (min_x, max_x, min_y, max_y, min_z, max_z) specifying 
                      the bounding box in voxel coordinates
        base_dir (str): Directory containing H5 files
        dataset_name (str): Name of the dataset inside each H5 file (default: 'data')
    
    Returns:
        numpy.ndarray: A 3D array containing the data within the specified bounding box
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    
    # Validate the bounding box
    if min_x > max_x or min_y > max_y or min_z > max_z:
        raise ValueError("Invalid bounding box: minimum values must be less than or equal to maximum values")
    
    # Calculate dimensions of the output array
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    depth = max_z - min_z + 1
    
    print(f"Loading data for bounding box: X[{min_x}-{max_x}], Y[{min_y}-{max_y}], Z[{min_z}-{max_z}]")
    print(f"Output dimensions: {width}x{height}x{depth}")
    
    # Find which files we need to load
    file_size = 1024  # Each file is 1024³ voxels
    
    # Calculate the file indices needed
    start_x_file = min_x // file_size
    end_x_file = max_x // file_size
    
    start_y_file = min_y // file_size
    end_y_file = max_y // file_size
    
    start_z_file = min_z // file_size
    end_z_file = max_z // file_size
    
    # Initialize the output array
    # Determine the data type by loading a sample file
    sample_file_name = f"x{start_x_file}y{start_y_file}z{start_z_file}.h5"
    sample_file_path = os.path.join(base_dir, sample_file_name)
    
    if not os.path.exists(sample_file_path):
        # Try to find any H5 file in the directory
        h5_files = list(Path(base_dir).glob("x*y*z*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {base_dir}")
        sample_file_path = str(h5_files[0])
    
    with h5py.File(sample_file_path, 'r') as f:
        if dataset_name not in f:
            # Try to find the correct dataset name
            if len(f.keys()) > 0:
                dataset_name = list(f.keys())[0]
                print(f"Dataset name '{dataset_name}' not found, using '{dataset_name}' instead")
            else:
                raise KeyError(f"Dataset not found in {sample_file_path}")
        
        dtype = f[dataset_name].dtype
    
    # Create the output array
    output_data = np.zeros((depth, height, width), dtype=dtype)
    
    # Track which files we loaded
    loaded_files = []
    start_time = time.time()
    
    # Load data from each required file
    for x_file in range(start_x_file, end_x_file + 1):
        for y_file in range(start_y_file, end_y_file + 1):
            for z_file in range(start_z_file, end_z_file + 1):
                file_name = f"x{x_file}y{y_file}z{z_file}.h5"
                file_path = os.path.join(base_dir, file_name)
                
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found, filling with zeros")
                    continue
                
                loaded_files.append(file_name)
                
                # Calculate the region of this file that intersects with our bounding box
                file_start_x = x_file * file_size
                file_start_y = y_file * file_size
                file_start_z = z_file * file_size
                
                # Calculate file coordinates relative to the bounding box
                rel_start_x = max(0, file_start_x - min_x)
                rel_start_y = max(0, file_start_y - min_y)
                rel_start_z = max(0, file_start_z - min_z)
                
                rel_end_x = min(width, rel_start_x + file_size - max(0, min_x - file_start_x))
                rel_end_y = min(height, rel_start_y + file_size - max(0, min_y - file_start_y))
                rel_end_z = min(depth, rel_start_z + file_size - max(0, min_z - file_start_z))
                
                # Calculate file coordinates
                file_min_x = max(0, min_x - file_start_x)
                file_min_y = max(0, min_y - file_start_y)
                file_min_z = max(0, min_z - file_start_z)
                
                file_max_x = min(file_size, file_min_x + (rel_end_x - rel_start_x))
                file_max_y = min(file_size, file_min_y + (rel_end_y - rel_start_y))
                file_max_z = min(file_size, file_min_z + (rel_end_z - rel_start_z))
                
                # Load the data
                with h5py.File(file_path, 'r') as f:
                    file_data = f[dataset_name][file_min_z:file_max_z, 
                                              file_min_y:file_max_y, 
                                              file_min_x:file_max_x]
                
                # Insert into the output array
                output_data[rel_start_z:rel_end_z, 
                           rel_start_y:rel_end_y, 
                           rel_start_x:rel_end_x] = file_data
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(loaded_files)} files in {elapsed_time:.2f} seconds")
    
    return output_data


# Example usage
if __name__ == "__main__":
    # Example bounding box: (min_x, max_x, min_y, max_y, min_z, max_z)
    example_bbox = (100, 600, 200, 700, 300, 800)
    
    # Test the function with the example bounding box
    try:
        data = load_h5_with_bbox(example_bbox)
        print(f"Successfully loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")