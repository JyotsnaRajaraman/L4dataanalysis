"""
PyTorch trainer script that closely replicates the original TensorFlow EM mask training
Supports both pickle coordinates and TFRecord (when converted), rotation augmentation, weighted training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import h5py
import pickle
import json
import os
import argparse
import logging
from pathlib import Path
import random
from typing import Tuple, List, Dict
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pad_concat(tensor1, tensor2):
    """
    Pad and concatenate tensors to handle size mismatches during upsampling.
    This replicates the pad_concat functionality from the original TensorFlow version.
    """
    # Get shapes
    shape1 = tensor1.shape[2:]  # [D, H, W]
    shape2 = tensor2.shape[2:]  # [D, H, W]
    
    # Calculate padding needed for tensor2 to match tensor1
    pad_d = shape1[0] - shape2[0]
    pad_h = shape1[1] - shape2[1] 
    pad_w = shape1[2] - shape2[2]
    
    # Apply padding if needed (pad format: [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_d_front = pad_d // 2
        pad_d_back = pad_d - pad_d_front
        
        tensor2 = F.pad(tensor2, [pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back])
    
    # Concatenate along channel dimension (dim=1)
    return torch.cat([tensor1, tensor2], dim=1)

class ConvRelu3D(nn.Module):
    """3D Convolution + ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(ConvRelu3D, self).__init__()
        
        # Handle padding
        if padding == 'same':
            # Calculate padding for 'same' behavior
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            padding = (pad_w, pad_h, pad_d)
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x))

class DTU2UNet3D(nn.Module):
    """
    DTU-2 UNet from:
    Synaptic Cleft Segmentation in Non-isotropic Volume Electron Microscopy of the
    Complete Drosophila Brain
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00934-2_36.pdf
    
    This is a faithful PyTorch port of the original TensorFlow implementation.
    """
    
    def __init__(self, in_channels=1, num_classes=1):
        super(DTU2UNet3D, self).__init__()
        
        # Encoder blocks
        # Block 1 - uses (1,3,3) kernels and (1,3,3) pooling
        self.conv1a = ConvRelu3D(in_channels, 12, (1, 3, 3))
        self.conv1b = ConvRelu3D(12, 12, (1, 3, 3))
        self.pool1 = nn.MaxPool3d((1, 3, 3))
        
        # Block 2 - uses (1,3,3) kernels and (1,3,3) pooling  
        self.conv2a = ConvRelu3D(12, 72, (1, 3, 3))
        self.conv2b = ConvRelu3D(72, 72, (1, 3, 3))
        self.pool2 = nn.MaxPool3d((1, 3, 3))
        
        # Block 3 - uses (3,3,3) kernels and (3,3,3) pooling
        self.conv3a = ConvRelu3D(72, 432, (3, 3, 3))
        self.conv3b = ConvRelu3D(432, 432, (3, 3, 3))
        self.pool3 = nn.MaxPool3d((3, 3, 3))
        
        # Block 4 - bottleneck with (3,3,3) kernels
        self.conv4a = ConvRelu3D(432, 2592, (3, 3, 3))
        self.conv4b = ConvRelu3D(2592, 2592, (3, 3, 3))
        
        # Decoder blocks
        # Block 5 - upsample with (3,3,3) and stride (3,3,3)
        self.upconv5 = nn.ConvTranspose3d(2592, 432, (3, 3, 3), stride=(3, 3, 3))
        self.conv5 = ConvRelu3D(432 + 432, 432, (3, 3, 3))  # 432 from skip + 432 from upconv
        
        # Block 6 - upsample with (1,3,3) and stride (1,3,3)
        self.upconv6 = nn.ConvTranspose3d(432, 72, (1, 3, 3), stride=(1, 3, 3))
        self.conv6 = ConvRelu3D(72 + 72, 72, (1, 3, 3))  # 72 from skip + 72 from upconv
        
        # Block 7 - upsample with (1,3,3) and stride (1,3,3)
        self.upconv7 = nn.ConvTranspose3d(72, 12, (1, 3, 3), stride=(1, 3, 3))
        self.conv7 = ConvRelu3D(12 + 12, 12, (1, 3, 3))  # 12 from skip + 12 from upconv
        
        # Final output layer
        self.conv8 = nn.Conv3d(12, num_classes, (1, 1, 1))
        
        self._debug_shapes = False
        
    def set_debug_shapes(self, debug=True):
        """Enable/disable shape debugging"""
        self._debug_shapes = debug
        
    def forward(self, x):
        if self._debug_shapes:
            logger.info(f'Input: {x.shape}')
            
        # Encoder path
        # Block 1
        conv1 = self.conv1a(x)
        conv1 = self.conv1b(conv1)
        if self._debug_shapes:
            logger.info(f'conv1: {conv1.shape}')
        pool1 = self.pool1(conv1)
        if self._debug_shapes:
            logger.info(f'pool1: {pool1.shape}')
        
        # Block 2  
        conv2 = self.conv2a(pool1)
        conv2 = self.conv2b(conv2)
        if self._debug_shapes:
            logger.info(f'conv2: {conv2.shape}')
        pool2 = self.pool2(conv2)
        if self._debug_shapes:
            logger.info(f'pool2: {pool2.shape}')
        
        # Block 3
        conv3 = self.conv3a(pool2)
        conv3 = self.conv3b(conv3)
        if self._debug_shapes:
            logger.info(f'conv3: {conv3.shape}')
        pool3 = self.pool3(conv3)
        if self._debug_shapes:
            logger.info(f'pool3: {pool3.shape}')
        
        # Block 4 (bottleneck)
        conv4 = self.conv4a(pool3)
        conv4 = self.conv4b(conv4)
        if self._debug_shapes:
            logger.info(f'conv4: {conv4.shape}')
        
        # Decoder path
        # Block 5
        upconv5 = F.relu(self.upconv5(conv4))
        if self._debug_shapes:
            logger.info(f'upconv5: {upconv5.shape}')
        concat5 = pad_concat(conv3, upconv5)
        if self._debug_shapes:
            logger.info(f'concat5: {concat5.shape}')
        conv5 = self.conv5(concat5)
        if self._debug_shapes:
            logger.info(f'conv5: {conv5.shape}')
        
        # Block 6
        upconv6 = F.relu(self.upconv6(conv5))
        if self._debug_shapes:
            logger.info(f'upconv6: {upconv6.shape}')
        concat6 = pad_concat(conv2, upconv6)
        if self._debug_shapes:
            logger.info(f'concat6: {concat6.shape}')
        conv6 = self.conv6(concat6)
        if self._debug_shapes:
            logger.info(f'conv6: {conv6.shape}')
        
        # Block 7
        upconv7 = F.relu(self.upconv7(conv6))
        if self._debug_shapes:
            logger.info(f'upconv7: {upconv7.shape}')
        concat7 = pad_concat(conv1, upconv7)
        if self._debug_shapes:
            logger.info(f'concat7: {concat7.shape}')
        conv7 = self.conv7(concat7)
        if self._debug_shapes:
            logger.info(f'conv7: {conv7.shape}')
        
        # Final output
        output = self.conv8(conv7)
        if self._debug_shapes:
            logger.info(f'output: {output.shape}')
        
        return output

def load_chunk_from_volume(coord, volume, chunk_shape):
    """Load a chunk from HDF5 volume at given coordinates"""
    coord = np.array(coord, dtype=np.int32)
    chunk_shape = np.array(chunk_shape, dtype=np.int32)

    # Calculate start and end positions
    start_offset = chunk_shape // 2
    starts = coord - start_offset
    ends = starts + chunk_shape

    # Handle boundaries
    vol_shape = np.array(volume.shape[:3], dtype=np.int32)
    starts = np.maximum(starts, 0)
    ends = np.minimum(ends, vol_shape)

    # Create output array
    if len(volume.shape) == 3:
        output = np.zeros(chunk_shape, dtype=np.float32)

        # Extract data
        data = volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

        # Calculate where to place the data in output
        data_start = starts - (coord - start_offset)
        data_end = data_start + (ends - starts)

        # Ensure indices are within bounds
        data_start = np.maximum(data_start, 0)
        data_end = np.minimum(data_end, chunk_shape)

        output[data_start[0]:data_end[0],
               data_start[1]:data_end[1],
               data_start[2]:data_end[2]] = data.astype(np.float32)

        # Add channel dimension [D, H, W, C]
        output = np.expand_dims(output, axis=-1)

    else:  # 4D volume with channels
        output = np.zeros(tuple(chunk_shape) + (volume.shape[-1],), dtype=np.float32)

        data = volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2], :]

        data_start = starts - (coord - start_offset)
        data_end = data_start + (ends - starts)

        data_start = np.maximum(data_start, 0)
        data_end = np.minimum(data_end, chunk_shape)

        output[data_start[0]:data_end[0],
               data_start[1]:data_end[1],
               data_start[2]:data_end[2], :] = data.astype(np.float32)

    return output

class EMDataset(Dataset):
    """PyTorch dataset for EM data with augmentation and weighting support"""

    def __init__(self,
                 data_volumes: str,
                 label_volumes: str,
                 coordinates: List[Tuple],
                 fov_size: Tuple[int, int, int],
                 label_size: Tuple[int, int, int],
                 image_mean: float = 128.0,
                 image_stddev: float = 33.0,
                 num_classes: int = 1,
                 rotation: bool = False,
                 weighted: bool = False,
                 weights_volumes: str = None):

        self.coordinates = coordinates
        self.fov_size = fov_size
        self.label_size = label_size
        self.image_mean = image_mean
        self.image_stddev = image_stddev
        self.num_classes = num_classes
        self.rotation = rotation
        self.weighted = weighted

        # Load volumes into memory
        logger.info("Loading image volumes...")
        self.image_volumes = {}
        for vol in data_volumes.split(','):
            volname, path, dataset = vol.split(':')
            try:
                with h5py.File(path, 'r') as f:
                    self.image_volumes[volname] = f[dataset][:]
                    logger.info(f"Loaded image volume {volname}: {self.image_volumes[volname].shape}, dtype: {self.image_volumes[volname].dtype}")
            except Exception as e:
                logger.error(f"Error loading image volume {volname} from {path}:{dataset} - {e}")
                raise

        logger.info("Loading label volumes...")
        self.label_volumes = {}
        for vol in label_volumes.split(','):
            volname, path, dataset = vol.split(':')
            try:
                with h5py.File(path, 'r') as f:
                    data = f[dataset][:]
                    if num_classes > 1:
                        # Convert to one-hot if multi-class
                        unique_labels = np.unique(data)
                        logger.info(f"Found labels: {unique_labels}")
                        data_one_hot = np.zeros(data.shape + (num_classes,), dtype=np.float32)
                        for i, label in enumerate(unique_labels[:num_classes]):
                            data_one_hot[..., i] = (data == label).astype(np.float32)
                        data = data_one_hot
                    else:
                        # Binary case - normalize to 0-1
                        data = (data > 0).astype(np.float32)
                    self.label_volumes[volname] = data
                    logger.info(f"Loaded label volume {volname}: {self.label_volumes[volname].shape}, dtype: {self.label_volumes[volname].dtype}")
            except Exception as e:
                logger.error(f"Error loading label volume {volname} from {path}:{dataset} - {e}")
                raise

        # Load weight volumes if specified
        self.weight_volumes = {}
        if weights_volumes and self.weighted:
            logger.info("Loading weight volumes...")
            for vol in weights_volumes.split(','):
                volname, path, dataset = vol.split(':')
                try:
                    with h5py.File(path, 'r') as f:
                        self.weight_volumes[volname] = f[dataset][:]
                        logger.info(f"Loaded weight volume {volname}: {self.weight_volumes[volname].shape}")
                except Exception as e:
                    logger.error(f"Error loading weight volume {volname} from {path}:{dataset} - {e}")
                    raise

        # Check that we have matching volume names
        coord_volumes = set([coord[1] for coord in coordinates[:100]])  # Check first 100
        available_image_volumes = set(self.image_volumes.keys())
        available_label_volumes = set(self.label_volumes.keys())

        logger.info(f"Coordinate volume names (sample): {coord_volumes}")
        logger.info(f"Available image volumes: {available_image_volumes}")
        logger.info(f"Available label volumes: {available_label_volumes}")

        if not coord_volumes.issubset(available_image_volumes):
            logger.warning(f"Missing image volumes: {coord_volumes - available_image_volumes}")
        if not coord_volumes.issubset(available_label_volumes):
            logger.warning(f"Missing label volumes: {coord_volumes - available_label_volumes}")

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coord, volname = self.coordinates[idx]

        try:
            # Map coordinate volume names to actual volume names
            # Your coordinates use 'validation1' but volumes might be named differently
            image_volname = volname
            label_volname = volname

            # Create a mapping for common mismatches
            volume_mapping = {
                'validation1': 'vol',  # Map validation1 to vol for images
                'vol': 'vol'
            }

            # Try to find a matching volume name for images
            if volname not in self.image_volumes:
                if volname in volume_mapping and volume_mapping[volname] in self.image_volumes:
                    image_volname = volume_mapping[volname]
                else:
                    # Use the first available image volume
                    available_image_vols = list(self.image_volumes.keys())
                    if available_image_vols:
                        image_volname = available_image_vols[0]
                        logger.debug(f"Using image volume {image_volname} for coordinate volume {volname}")
                    else:
                        raise KeyError(f"No image volumes available")

            if volname not in self.label_volumes:
                # Use the first available label volume
                available_label_vols = list(self.label_volumes.keys())
                if available_label_vols:
                    label_volname = available_label_vols[0]
                else:
                    raise KeyError(f"No label volumes available")

            # Load image chunk
            image_vol = self.image_volumes[image_volname]
            image = load_chunk_from_volume(coord, image_vol, self.fov_size)

            # Load label chunk
            label_vol = self.label_volumes[label_volname]
            label = load_chunk_from_volume(coord, label_vol, self.label_size)

            # Load weight chunk if available
            weight = None
            if self.weighted and volname in self.weight_volumes:
                weight_vol = self.weight_volumes[volname]
                weight = load_chunk_from_volume(coord, weight_vol, self.label_size)
                weight = np.transpose(weight, (3, 0, 1, 2))
                weight = torch.from_numpy(weight).float()

            # Preprocess image
            image = (image - self.image_mean) / self.image_stddev

            # Convert to PyTorch format [C, D, H, W]
            image = np.transpose(image, (3, 0, 1, 2))  # [C, D, H, W]
            label = np.transpose(label, (3, 0, 1, 2))   # [C, D, H, W]

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            # Apply rotation augmentation if enabled
            if self.rotation:
                image, label = apply_rotation_augmentation(image, label, self.rotation)
                if weight is not None:
                    # Apply same augmentation to weights (simplified)
                    weight, _ = apply_rotation_augmentation(weight, weight, self.rotation)

            result = {
                'image': image,
                'label': label,
                'center': torch.tensor(coord, dtype=torch.long),
                'volname': volname
            }

            if weight is not None:
                result['weight'] = weight

            return result

        except Exception as e:
            logger.error(f"Error loading sample {idx}, coord {coord}, volume {volname}: {e}")
            # Return a dummy sample to avoid crashing
            dummy_image = torch.zeros((1,) + self.fov_size, dtype=torch.float32)
            dummy_label = torch.zeros((1,) + self.label_size, dtype=torch.float32)
            result = {
                'image': dummy_image,
                'label': dummy_label,
                'center': torch.tensor([0, 0, 0], dtype=torch.long),
                'volname': 'dummy'
            }
            if self.weighted:
                result['weight'] = torch.ones_like(dummy_label)
            return result

def apply_rotation_augmentation(image, label, rotation=True):
    """Apply rotation augmentation similar to the original TensorFlow implementation"""
    if not rotation:
        return image, label
    
    # Random rotation around each axis (90 degree increments)
    axes_pairs = [(2, 3), (2, 4), (3, 4)]  # (H,W), (H,D), (W,D) for CDHW format
    
    for axes in axes_pairs:
        if random.random() < 0.25:  # 25% chance for each rotation
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            image = torch.rot90(image, k=k, dims=axes)
            label = torch.rot90(label, k=k, dims=axes)
    
    # Random flips
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])  # flip D
        label = torch.flip(label, dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[3])  # flip H  
        label = torch.flip(label, dims=[3])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[4])  # flip W
        label = torch.flip(label, dims=[4])
        
    return image, label

def calculate_class_weights(labels, num_classes):
    """Calculate class weights for weighted loss similar to TensorFlow implementation"""
    if num_classes == 1:
        # Binary case - calculate pos/neg weights
        pos_count = torch.sum(labels > 0.5).float()
        neg_count = torch.sum(labels <= 0.5).float()
        total = pos_count + neg_count
        
        if pos_count > 0 and neg_count > 0:
            pos_weight = total / (2.0 * pos_count)
            neg_weight = total / (2.0 * neg_count)
            return torch.tensor([neg_weight, pos_weight])
        else:
            return torch.tensor([1.0, 1.0])
    else:
        # Multi-class case
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            class_count = torch.sum(labels == i).float()
            if class_count > 0:
                weights[i] = len(labels) / (num_classes * class_count)
            else:
                weights[i] = 1.0
        return weights

def match_tensor_sizes(output, target, debug=False):
    """
    Match the spatial dimensions of output and target tensors by cropping or padding.
    This handles the size mismatches caused by the DTU-2 UNet architecture.
    """
    output_shape = output.shape[2:]  # [D, H, W]
    target_shape = target.shape[2:]  # [D, H, W]
    
    if debug:
        logger.info(f"match_tensor_sizes: output {output_shape} -> target {target_shape}")
    
    if output_shape == target_shape:
        return output, target
    
    # We need to make both tensors the same size
    # Strategy: crop to the minimum size in each dimension
    min_d = min(output_shape[0], target_shape[0])
    min_h = min(output_shape[1], target_shape[1])
    min_w = min(output_shape[2], target_shape[2])
    
    if debug:
        logger.info(f"Target dimensions: D={min_d}, H={min_h}, W={min_w}")
    
    # Crop output if needed
    if output_shape != (min_d, min_h, min_w):
        start_d = (output_shape[0] - min_d) // 2
        start_h = (output_shape[1] - min_h) // 2
        start_w = (output_shape[2] - min_w) // 2
        
        end_d = start_d + min_d
        end_h = start_h + min_h
        end_w = start_w + min_w
        
        if debug:
            logger.info(f"Cropping output from {output_shape} to {(min_d, min_h, min_w)}")
            logger.info(f"Output crop: D[{start_d}:{end_d}], H[{start_h}:{end_h}], W[{start_w}:{end_w}]")
        
        output = output[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
    
    # Crop target if needed
    if target_shape != (min_d, min_h, min_w):
        start_d = (target_shape[0] - min_d) // 2
        start_h = (target_shape[1] - min_h) // 2
        start_w = (target_shape[2] - min_w) // 2
        
        end_d = start_d + min_d
        end_h = start_h + min_h
        end_w = start_w + min_w
        
        if debug:
            logger.info(f"Cropping target from {target_shape} to {(min_d, min_h, min_w)}")
            logger.info(f"Target crop: D[{start_d}:{end_d}], H[{start_h}:{end_h}], W[{start_w}:{end_w}]")
        
        target = target[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
    
    if debug:
        logger.info(f"Final shapes: output {output.shape[2:]}, target {target.shape[2:]}")
    
    return output, target
    """
    Match the spatial dimensions of output and target tensors by cropping or padding.
    This handles the size mismatches caused by the DTU-2 UNet architecture.
    """
    output_shape = output.shape[2:]  # [D, H, W]
    target_shape = target.shape[2:]  # [D, H, W]
    
    if output_shape == target_shape:
        return output, target
    
    # Calculate differences
    diff_d = output_shape[0] - target_shape[0]
    diff_h = output_shape[1] - target_shape[1] 
    diff_w = output_shape[2] - target_shape[2]
    
    # If output is larger, crop it to match target
    if diff_d > 0 or diff_h > 0 or diff_w > 0:
        start_d = max(0, diff_d // 2)
        start_h = max(0, diff_h // 2)
        start_w = max(0, diff_w // 2)
        
        end_d = start_d + target_shape[0]
        end_h = start_h + target_shape[1]
        end_w = start_w + target_shape[2]
        
        output = output[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
    
    # If output is smaller, pad it to match target
    elif diff_d < 0 or diff_h < 0 or diff_w < 0:
        pad_d_before = abs(diff_d) // 2 if diff_d < 0 else 0
        pad_d_after = abs(diff_d) - pad_d_before if diff_d < 0 else 0
        pad_h_before = abs(diff_h) // 2 if diff_h < 0 else 0
        pad_h_after = abs(diff_h) - pad_h_before if diff_h < 0 else 0
        pad_w_before = abs(diff_w) // 2 if diff_w < 0 else 0
        pad_w_after = abs(diff_w) - pad_w_before if diff_w < 0 else 0
        
        output = F.pad(output, [pad_w_before, pad_w_after, 
                               pad_h_before, pad_h_after,
                               pad_d_before, pad_d_after])
    
    return output, target
    """Create orthogonal projection for visualization"""
    if len(volume_tensor.shape) == 4:  # [C, D, H, W]
        # Take mean along depth (D) dimension
        projection = torch.mean(volume_tensor[0], dim=0)  # [H, W]
        return projection.unsqueeze(0)  # [1, H, W] for CHW format
    elif len(volume_tensor.shape) == 5:  # [B, C, D, H, W]
        projection = torch.mean(volume_tensor[0, 0], dim=0)  # [H, W]
        return projection.unsqueeze(0)  # [1, H, W] for CHW format
    return volume_tensor

class PurePyTorchTrainer:
    """Pure PyTorch trainer that replicates TensorFlow EM mask training functionality"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Initialize DTU-2 UNet model
        self.model = DTU2UNet3D(
            in_channels=1,
            num_classes=args.num_classes
        ).to(self.device)
        
        # Enable debug shapes for first few steps
        self.model.set_debug_shapes(True)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"DTU-2 UNet parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Initialize loss function based on training type
        if args.weighted:
            if args.num_classes == 1:
                # Binary weighted loss
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                # Multi-class weighted loss
                self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            if args.num_classes == 1:
                # Use BCEWithLogitsLoss instead of MSE for binary classification
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

        # Initialize TensorBoard
        self.writer = SummaryWriter(args.train_dir)

    def create_dataloader(self):
        """Create PyTorch dataloader"""
        # Load coordinates from pickle
        logger.info(f"Loading coordinates from {self.args.coordinates_pkl}")
        with open(self.args.coordinates_pkl, 'rb') as f:
            all_coordinates = pickle.load(f)

        logger.info(f"Loaded {len(all_coordinates):,} coordinates")

        # Subsample if requested
        if hasattr(self.args, 'max_samples') and self.args.max_samples > 0:
            if self.args.max_samples < len(all_coordinates):
                coordinates = random.sample(all_coordinates, self.args.max_samples)
                logger.info(f"Subsampled to {len(coordinates):,} coordinates")
            else:
                coordinates = all_coordinates
        else:
            coordinates = all_coordinates

        # Create dataset
        dataset = EMDataset(
            data_volumes=self.args.data_volumes,
            label_volumes=self.args.label_volumes,
            coordinates=coordinates,
            fov_size=self.args.fov_size,
            label_size=self.args.label_size,
            image_mean=self.args.image_mean,
            image_stddev=self.args.image_stddev,
            num_classes=self.args.num_classes,
            rotation=self.args.rotation,
            weighted=self.args.weighted,
            weights_volumes=getattr(self.args, 'weights_volumes', None)
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,  # Reduced for stability
            pin_memory=True,
            drop_last=True
        )

        return dataloader

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        images = batch['image'].to(self.device, non_blocking=True)
        labels = batch['label'].to(self.device, non_blocking=True)
        weights = batch.get('weight', None)
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(images)
        
        # Debug logging for first few steps - initialize step count first
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        # Handle size mismatches between output and labels
        debug_sizing = self._step_count < 3
        outputs, labels = match_tensor_sizes(outputs, labels, debug=debug_sizing)
        if weights is not None:
            weights, _ = match_tensor_sizes(weights, labels, debug=False)
            
        if self._step_count < 5:
            logger.info(f"Step {self._step_count}: Output shape: {outputs.shape}, Label shape: {labels.shape}")

        # Calculate loss
        if self.args.num_classes == 1:
            # Binary classification
            if self.args.weighted and weights is not None:
                loss = self.criterion(outputs, labels)
                loss = loss * weights
                loss = loss.mean()
            else:
                loss = self.criterion(outputs, labels)
        else:
            # Multi-class classification
            labels_idx = torch.argmax(labels, dim=1)  # [B, D, H, W]
            if self.args.weighted and weights is not None:
                loss = self.criterion(outputs, labels_idx.long())
                loss = loss * weights.squeeze(1)  # Remove channel dim from weights
                loss = loss.mean()
            else:
                loss = self.criterion(outputs, labels_idx.long())

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item(), outputs

    def train(self):
        """Main training loop"""
        dataloader = self.create_dataloader()

        step = 0
        epoch = 0

        logger.info("Starting training...")
        logger.info(f"Dataloader length: {len(dataloader)}")
        logger.info(f"Total steps planned: {self.args.max_steps}")

        try:
            while step < self.args.max_steps:
                for batch in dataloader:
                    if step >= self.args.max_steps:
                        break

                    try:
                        loss, outputs = self.train_step(batch)

                        # Logging
                        if step % 100 == 0:
                            logger.info(f"Step {step:6d}, Loss: {loss:.6f}")
                            self.writer.add_scalar('Loss/Train', loss, step)

                            # Log learning rate
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.writer.add_scalar('Learning_Rate', current_lr, step)

                        # Disable debug shapes after first few steps
                        if step == 3:
                            self.model.set_debug_shapes(False)
                        # Visualization
                        if step % 500 == 0:
                            self._log_images(batch, outputs, step)

                        # Save checkpoint
                        if step % 1000 == 0 and step > 0:
                            self.save_checkpoint(step)

                        step += 1

                    except Exception as e:
                        logger.error(f"Error in training step {step}: {e}")
                        step += 1
                        continue

                epoch += 1
                logger.info(f"Completed epoch {epoch}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        # Save final checkpoint
        self.save_checkpoint(step, final=True)
        self.writer.close()
        logger.info("Training completed!")

    def _log_images(self, batch, outputs, step):
        """Log images to TensorBoard"""
        try:
            with torch.no_grad():
                images = batch['image'][:1]  # First item
                labels = batch['label'][:1]
                preds = outputs[:1]

                # Create projections
                #img_proj = create_orthogonal_projection(images)
                #label_proj = create_orthogonal_projection(labels)
                #pred_proj = create_orthogonal_projection(preds)

                # Normalize for visualization
                #img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
                #label_proj = (label_proj - label_proj.min()) / (label_proj.max() - label_proj.min() + 1e-8)
                #pred_proj = torch.sigmoid(pred_proj)

                #self.writer.add_image('Image/Projection', img_proj, step)
                #self.writer.add_image('Label/Projection', label_proj, step)
                #self.writer.add_image('Prediction/Projection', pred_proj, step)

        except Exception as e:
            logger.warning(f"Error logging images at step {step}: {e}")

    def save_checkpoint(self, step, final=False):
        """Save checkpoint"""
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'args': vars(self.args)
            }

            suffix = 'final' if final else f'{step:06d}'
            checkpoint_path = os.path.join(self.args.train_dir, f'checkpoint_{suffix}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}").argmax(labels, dim=1)  # [B, D, H, W]
            loss = self.criterion(outputs, labels.long())

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item(), outputs

    def train(self):
        """Main training loop"""
        dataloader = self.create_dataloader()

        step = 0
        epoch = 0

        logger.info("Starting training...")
        logger.info(f"Dataloader length: {len(dataloader)}")
        logger.info(f"Total steps planned: {self.args.max_steps}")

        try:
            while step < self.args.max_steps:
                for batch in dataloader:
                    if step >= self.args.max_steps:
                        break

                    try:
                        loss, outputs = self.train_step(batch)

                        # Logging
                        if step % 100 == 0:
                            logger.info(f"Step {step:6d}, Loss: {loss:.6f}")
                            self.writer.add_scalar('Loss/Train', loss, step)

                            # Log learning rate
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.writer.add_scalar('Learning_Rate', current_lr, step)

                        # Disable debug shapes after first few steps
                        if step == 3:
                            self.model.set_debug_shapes(False)
                        if step % 500 == 0:
                            self._log_images(batch, outputs, step)

                        # Save checkpoint
                        if step % 1000 == 0 and step > 0:
                            self.save_checkpoint(step)

                        step += 1

                    except Exception as e:
                        logger.error(f"Error in training step {step}: {e}")
                        step += 1
                        continue

                epoch += 1
                logger.info(f"Completed epoch {epoch}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        # Save final checkpoint
        self.save_checkpoint(step, final=True)
        self.writer.close()
        logger.info("Training completed!")

    def _log_images(self, batch, outputs, step):
        """Log images to TensorBoard"""
        try:
            with torch.no_grad():
                images = batch['image'][:1]  # First item
                labels = batch['label'][:1]
                preds = outputs[:1]

                # Create projections
                img_proj = create_orthogonal_projection(images)
                label_proj = create_orthogonal_projection(labels)
                pred_proj = create_orthogonal_projection(preds)

                # Normalize for visualization
                img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
                label_proj = (label_proj - label_proj.min()) / (label_proj.max() - label_proj.min() + 1e-8)
                pred_proj = torch.sigmoid(pred_proj)

                self.writer.add_image('Image/Projection', img_proj, step)
                self.writer.add_image('Label/Projection', label_proj, step)
                self.writer.add_image('Prediction/Projection', pred_proj, step)

        except Exception as e:
            logger.warning(f"Error logging images at step {step}: {e}")

    def save_checkpoint(self, step, final=False):
        """Save checkpoint"""
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'args': vars(self.args)
            }

            suffix = 'final' if final else f'{step:06d}'
            checkpoint_path = os.path.join(self.args.train_dir, f'checkpoint_{suffix}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Pure PyTorch EM mask training with DTU-2 UNet')

    parser.add_argument('--data_volumes', type=str, required=True)
    parser.add_argument('--label_volumes', type=str, required=True)
    parser.add_argument('--weights_volumes', type=str, default=None,
                       help='Comma-separated list of weight volumes')
    parser.add_argument('--coordinates_pkl', type=str, required=True,
                       help='Path to pickle file with coordinates')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--model_args', type=str, required=True)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_mean', type=float, default=128)
    parser.add_argument('--image_stddev', type=float, default=33)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--max_samples', type=int, default=100000,
                       help='Maximum number of coordinate samples to use (0 = use all)')
    parser.add_argument('--rotation', action='store_true', default=False,
                       help='Enable rotation augmentation')
    parser.add_argument('--weighted', action='store_true', default=False,
                       help='Use weighted loss function')

    return parser.parse_args()

def main():
    args = parse_args()

    # Parse model arguments
    model_args = json.loads(args.model_args)
    args.fov_size = tuple([int(i) for i in model_args['fov_size']])

    if 'label_size' in model_args:
        args.label_size = tuple([int(i) for i in model_args['label_size']])
    else:
        args.label_size = args.fov_size

    args.num_classes = int(model_args['num_classes'])

    # Create training directory
    os.makedirs(args.train_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Initialize trainer and start training
    trainer = PurePyTorchTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()

