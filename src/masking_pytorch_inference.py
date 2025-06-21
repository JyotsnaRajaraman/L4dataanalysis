"""
PyTorch inference script for trained EM mask model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import os
import argparse
import logging
from pathlib import Path
import itertools
from typing import Tuple, List, Dict
from mpi4py import MPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def pad_concat(tensor1, tensor2):
    """
    handle size mismatches during upsampling
    """
    # Get shapes
    shape1 = tensor1.shape[2:]  # [D, H, W]
    shape2 = tensor2.shape[2:]  # [D, H, W]
    print(f'{shape1=}')
    print(f'{shape2=}')
    if (shape1 >= shape2):
        print(f'shape 1 is greater than 2 as expected')
    else:
        print(f'WARNING: CHECK THIS shape 1 is NOT greater than 2 as expected')

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

        if padding == 'same':
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            # padding = (pad_w, pad_h, pad_d) # why this ordeR?? 
            padding = (pad_d, pad_h, pad_w)

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

    def forward(self, x):

        input_shape = x.shape[2:]
        print(f"UNet Input shape: {input_shape}")

        # Encoder path
        # Block 1
        conv1 = self.conv1a(x)
        conv1 = self.conv1b(conv1)
        pool1 = self.pool1(conv1)

        # Block 2
        conv2 = self.conv2a(pool1)
        conv2 = self.conv2b(conv2)
        pool2 = self.pool2(conv2)

        # Block 3
        conv3 = self.conv3a(pool2)
        conv3 = self.conv3b(conv3)
        pool3 = self.pool3(conv3)

        # Block 4 (bottleneck)
        conv4 = self.conv4a(pool3)
        conv4 = self.conv4b(conv4)

        # Decoder path
        # Block 5
        upconv5 = F.relu(self.upconv5(conv4))
        concat5 = pad_concat(conv3, upconv5)
        conv5 = self.conv5(concat5)

        # Block 6
        upconv6 = F.relu(self.upconv6(conv5))
        concat6 = pad_concat(conv2, upconv6)
        conv6 = self.conv6(concat6)

        # Block 7
        upconv7 = F.relu(self.upconv7(conv6))
        concat7 = pad_concat(conv1, upconv7)
        conv7 = self.conv7(concat7)

        # Final output
        output = self.conv8(conv7)

        output_shape = output.shape[2:]
        print(f"UNet Output shape: {output_shape}")
        print(f"Size difference: {np.array(output_shape) - np.array(input_shape)}")

        return output

def match_tensor_sizes(output, target_shape):
    """
    Crop output tensor to match target shape for inference
    """
    output_shape = output.shape[2:]  # [D, H, W]

    if output_shape == target_shape:
        return output

    # Crop to target size
    min_d = min(output_shape[0], target_shape[0])
    min_h = min(output_shape[1], target_shape[1])
    min_w = min(output_shape[2], target_shape[2])

    # Crop output if needed
    start_d = (output_shape[0] - min_d) // 2
    start_h = (output_shape[1] - min_h) // 2
    start_w = (output_shape[2] - min_w) // 2

    end_d = start_d + min_d
    end_h = start_h + min_h
    end_w = start_w + min_w

    output = output[:, :, start_d:end_d, start_h:end_h, start_w:end_w]

    return output

def load_model(checkpoint_path, num_classes=1):
    """Load trained PyTorch DTU-2 UNet model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DTU2UNet3D(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded DTU-2 UNet model from {checkpoint_path}")
    logger.info(f"Model trained for {checkpoint.get('step', 'unknown')} steps")

    return model, device

def get_h5_shape(volume_path):
    if ':' in volume_path:
        path, dataset = volume_path.split(':')
    else:
        path, dataset = volume_path, 'data'

    with h5py.File(path, 'r') as f:
        return f[dataset].shape

def load_chunk_from_h5(volume_path, coord, chunk_shape):
    """Load a chunk from HDF5 volume at given coordinates"""
    if ':' in volume_path:
        path, dataset = volume_path.split(':')
    else:
        path, dataset = volume_path, 'data'

    coord = np.array(coord, dtype=np.int32)
    chunk_shape = np.array(chunk_shape, dtype=np.int32)

    with h5py.File(path, 'r') as f:
        volume = f[dataset]
        vol_shape = np.array(volume.shape[:3], dtype=np.int32)

        start_offset = chunk_shape // 2
        starts = coord - start_offset
        ends = starts + chunk_shape

        # boundary limits
        starts = np.maximum(starts, 0)
        ends = np.minimum(ends, vol_shape)

        output = np.zeros(chunk_shape, dtype=np.float32)
        data = volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

        # data in output
        data_start = starts - (coord - start_offset)
        data_end = data_start + (ends - starts)

        # boundary limits
        data_start = np.maximum(data_start, 0)
        data_end = np.minimum(data_end, chunk_shape)

        output[data_start[0]:data_end[0],
               data_start[1]:data_end[1],
               data_start[2]:data_end[2]] = data.astype(np.float32)

    return output

# def generate_prediction_coordinates(input_offset, input_size, chunk_shape, overlap):
#     """Generate coordinates for prediction patches"""
#     input_offset = np.array(input_offset)
#     input_size = np.array(input_size)
#     chunk_shape = np.array(chunk_shape)
#     overlap = np.array(overlap)

#     # Calculate step size
#     step_size = chunk_shape - overlap

#     # Calculate number of steps in each dimension
#     steps = np.ceil((input_size - overlap) / step_size).astype(int)

#     # Generate coordinate grid
#     coordinates = []
#     for z in range(steps[0]):
#         for y in range(steps[1]):
#             for x in range(steps[2]):
#                 coord = input_offset + np.array([z, y, x]) * step_size + chunk_shape // 2

#                 # Ensure coordinates are within bounds
#                 coord = np.minimum(coord, input_offset + input_size - chunk_shape // 2 - 1)
#                 coord = np.maximum(coord, input_offset + chunk_shape // 2)

#                 coordinates.append(coord)

#     return coordinates

# def generate_prediction_coordinates2(input_offset, input_size, chunk_shape, overlap):
    input_offset = np.array(input_offset)
    input_size = np.array(input_size)
    chunk_shape = np.array(chunk_shape)
    overlap = np.array(overlap)

    print(f'inside gen coords: {input_offset=}, {input_size=}, {chunk_shape=}, {overlap=}')

    step_size = chunk_shape - overlap
    
    start_center = input_offset + chunk_shape // 2
    end_center = input_offset + input_size - chunk_shape // 2
    
    print(f'{step_size=}, {start_center=}, {end_center=}')
    
    coordinates = []
    
    for dim in range(3):
        if step_size[dim] <= 0:
            dim_coords = [start_center[dim]]
        else:
            # generate coordinates from start to end
            dim_coords = list(range(start_center[dim], end_center[dim] + 1, step_size[dim]))
            
            # end convered check
            if len(dim_coords) == 0 or dim_coords[-1] < end_center[dim]:
                dim_coords.append(end_center[dim])
        
        if dim == 0:
            z_coords = dim_coords
        elif dim == 1:
            y_coords = dim_coords
        else:
            x_coords = dim_coords
    
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                coordinates.append([z, y, x])
    
    return coordinates

def generate_prediction_coordinates_adjusted(input_offset, input_size, chunk_shape, 
                                           actual_output_shape, overlap):
    """
    Generate coordinates accounting for the actual output shape from the UNet
    """
    input_offset = np.array(input_offset)
    input_size = np.array(input_size)
    chunk_shape = np.array(chunk_shape)  # What we feed to the model
    actual_output_shape = np.array(actual_output_shape)  # What the model actually outputs
    overlap = np.array(overlap)
    
    print(f'Adjusted coord gen:')
    print(f'  Input chunk: {chunk_shape}')
    print(f'  Actual output: {actual_output_shape}')
    print(f'  Overlap: {overlap}')
    
    # Step size based on actual output size, not input size
    effective_step_size = actual_output_shape - overlap
    
    print(f'  Effective step size: {effective_step_size}')
    
    # Calculate the range we need to cover in terms of output coordinates
    # We need to think about where the output patches will be placed
    output_start = input_offset
    output_end = input_offset + input_size
    
    coordinates = []
    
    # Generate coordinates based on where outputs will be placed
    for dim in range(3):
        if effective_step_size[dim] <= 0:
            # If overlap >= output size, just use center coordinate
            dim_coords = [input_offset[dim] + chunk_shape[dim] // 2]
        else:
            # Start from first valid position
            first_center = input_offset[dim] + chunk_shape[dim] // 2
            
            # Generate positions where output patches will align properly
            dim_coords = []
            
            # The first patch center
            current_output_start = input_offset[dim]
            current_center = first_center
            
            while current_output_start < output_end[dim]:
                if current_center + chunk_shape[dim] // 2 <= input_offset[dim] + input_size[dim]:
                    dim_coords.append(int(current_center))
                
                # Move to next position based on effective step size
                current_output_start += effective_step_size[dim]
                # But the input center needs to account for the size difference
                size_diff = chunk_shape[dim] - actual_output_shape[dim]
                current_center = current_output_start + chunk_shape[dim] // 2
        
        if dim == 0:
            z_coords = dim_coords
        elif dim == 1:
            y_coords = dim_coords
        else:
            x_coords = dim_coords
    
    print(f'  Z coords: {len(z_coords)} positions')
    print(f'  Y coords: {len(y_coords)} positions') 
    print(f'  X coords: {len(x_coords)} positions')
    
    # Create all combinations
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                coordinates.append([z, y, x])
    
    return coordinates

def predict_batch(model, device, batch_coords, input_volume, chunk_shape,
                  image_mean=128.0, image_stddev=33.0):
    """Predict on a batch of coordinates"""
    batch_images = []

    for coord in batch_coords:
        image = load_chunk_from_h5(input_volume, coord, chunk_shape)

        # Preprocess
        image = (image - image_mean) / image_stddev

        # Add channel dimension and convert to tensor
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        batch_images.append(image)

    # Stack into batch [B, C, D, H, W]
    batch_tensor = torch.from_numpy(np.stack(batch_images)).float().to(device)

    print(f"Input tensor shape to model: {batch_tensor.shape}")
    print(f"Expected chunk shape: {chunk_shape}")


    with torch.no_grad():
        logits = model(batch_tensor)

        print(f"Raw model output shape: {logits.shape}")
        print(f"Raw output spatial dims: {logits.shape[2:]}")

        # UNet may produce different output shapes than input, crop to match the expected shape if needed
        target_shape = tuple(chunk_shape)
        if logits.shape[2:] != target_shape:
            print(f"mismatch in size in predict_batch")
            print(f"  Expected: {target_shape}")
            print(f"  Got:      {logits.shape[2:]}")
            print(f"  Difference: {np.array(logits.shape[2:]) - np.array(target_shape)}")

            logits = match_tensor_sizes(logits, target_shape)

            print(f"  After cropping: {logits.shape[2:]}")


        # Apply sigmoid for binary classification
        if logits.shape[1] == 1:
            predictions = torch.sigmoid(logits)
        else:
            predictions = torch.softmax(logits, dim=1)

        logits_np = logits.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

    return batch_coords, logits_np, predictions_np

def write_predictions_to_h5(output_volume, predictions_data, output_size,
                           chunk_shape, overlap, num_classes):
    """Write predictions to HDF5 file using MPI"""

    if ':' in output_volume:
        output_path, output_dataset = output_volume.split(':')
    else:
        output_path, output_dataset = output_volume, 'predictions'

    # Create output file (only rank 0)
    if mpi_rank == 0:
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            f.create_dataset('predictions', shape=output_size, dtype=np.float32)
            f.create_dataset('logits', shape=tuple(output_size) + (num_classes,), dtype=np.float32)
            logger.info(f"Created output file: {output_path}")
            logger.info(f"Output shape: {output_size}")

    mpi_comm.Barrier()

    chunk_shape = np.array(chunk_shape)
    overlap = np.array(overlap)

    with h5py.File(output_path, 'r+', driver='mpio', comm=mpi_comm) as f:
        pred_dataset = f['predictions']
        logits_dataset = f['logits']

        for batch_coords, batch_logits, batch_predictions in predictions_data:
            for coord, logits, prediction in zip(batch_coords, batch_logits, batch_predictions):
                coord = np.array(coord)

                actual_pred_shape = np.array(prediction.shape[1:])  # Skip channel dimension

                start_offset = actual_pred_shape // 2
                write_start = coord - start_offset + overlap // 2
                write_end = coord + start_offset - overlap // 2

                write_start = np.maximum(write_start, 0)
                write_end = np.minimum(write_end, output_size)

                write_region_shape = write_end - write_start

                pred_start = (actual_pred_shape - write_region_shape) // 2
                pred_end = pred_start + write_region_shape

                pred_start = np.maximum(pred_start, 0)
                pred_end = np.minimum(pred_end, actual_pred_shape)

                actual_write_size = pred_end - pred_start
                write_end = write_start + actual_write_size

                try:
                    if prediction.shape[0] == 1:  
                        pred_slice = prediction[0,
                                               pred_start[0]:pred_end[0],
                                               pred_start[1]:pred_end[1],
                                               pred_start[2]:pred_end[2]]

                        pred_dataset[write_start[0]:write_end[0],
                                    write_start[1]:write_end[1],
                                    write_start[2]:write_end[2]] = pred_slice

                    logits_slice = np.transpose(logits, (1, 2, 3, 0))[
                                   pred_start[0]:pred_end[0],
                                   pred_start[1]:pred_end[1],
                                   pred_start[2]:pred_end[2], :]

                    logits_dataset[write_start[0]:write_end[0],
                                  write_start[1]:write_end[1],
                                  write_start[2]:write_end[2], :] = logits_slice

                except Exception as e:
                    logger.warning(f"Error writing prediction at coord {coord}: {e}")
                    logger.warning(f"Prediction shape: {prediction.shape}, Write region: {write_start} to {write_end}")
                    logger.warning(f"Read region: {pred_start} to {pred_end}")
                    continue

class PyTorchInference:

    def __init__(self, args):
        self.args = args

        # Load model
        self.model, self.device = load_model(args.model_checkpoint, args.num_classes)

        # input parameters
        if args.input_offset and args.input_size:
            self.input_offset = [int(i) for i in args.input_offset.split(',')]
            self.input_size = [int(i) for i in args.input_size.split(',')]
        else:
            self.input_offset = [0, 0, 0]
            self.input_size = list(get_h5_shape(args.input_volume))

        self.chunk_shape = args.fov_size
        self.overlap = args.overlap if args.overlap else [0, 0, 0]

        # # coordinates for prediction
        # all_coordinates = generate_prediction_coordinates(
        #     self.input_offset, self.input_size, self.chunk_shape, self.overlap
        # )

        # actual output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, *self.chunk_shape).to(self.device)
            dummy_output = self.model(dummy_input)
            self.actual_output_shape = dummy_output.shape[2:]
        
        print(f"Input chunk shape: {self.chunk_shape}")
        print(f"Actual output shape: {self.actual_output_shape}")
        
        # Generate coordinates accounting for actual output size
        all_coordinates = generate_prediction_coordinates_adjusted(
            self.input_offset, self.input_size, self.chunk_shape, 
            self.actual_output_shape, self.overlap
        )

        coords_per_rank = len(all_coordinates) // mpi_size
        start_idx = mpi_rank * coords_per_rank

        if mpi_rank == mpi_size - 1: 
            end_idx = len(all_coordinates)
        else:
            end_idx = start_idx + coords_per_rank

        self.coordinates = all_coordinates[start_idx:end_idx]

        if mpi_rank == 0:
            logger.info(f"Total coordinates: {len(all_coordinates)}")
            logger.info(f"Input shape: {self.input_size}")
            logger.info(f"Chunk shape: {self.chunk_shape}")
            logger.info(f"Overlap: {self.overlap}")

        logger.info(f"Rank {mpi_rank}: Processing {len(self.coordinates)} coordinates")

    def run_inference(self):
        """Run inference on assigned coordinates"""
        predictions_data = []

        # Process in batches
        batch_size = self.args.batch_size
        num_batches = (len(self.coordinates) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(self.coordinates))
            batch_coords = self.coordinates[start_idx:end_idx]

            # Predict batch
            coords, logits, predictions = predict_batch(
                self.model, self.device, batch_coords,
                self.args.input_volume, self.chunk_shape,
                self.args.image_mean, self.args.image_stddev
            )

            predictions_data.append((coords, logits, predictions))

            if mpi_rank == 0 and (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{num_batches} batches")

        if mpi_rank == 0:
            logger.info("Writing predictions to output file...")

        write_predictions_to_h5(
            self.args.output_volume, predictions_data,
            self.input_size, self.chunk_shape, self.overlap,
            self.args.num_classes
        )

        if mpi_rank == 0:
            logger.info("Inference completed!")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch EM mask inference with DTU-2 UNet')

    parser.add_argument('--input_volume', type=str, required=True,
                       help='Path to input HDF5 volume (path:dataset)')
    parser.add_argument('--output_volume', type=str, required=True,
                       help='Path to output HDF5 volume (path:dataset)')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--model_args', type=str, required=True,
                       help='JSON string with model arguments')

    parser.add_argument('--input_offset', type=str, default='',
                       help='Input offset as x,y,z')
    parser.add_argument('--input_size', type=str, default='',
                       help='Input size as x,y,z')
    parser.add_argument('--overlap', type=int, nargs=3, default=[0, 0, 0],
                       help='Overlap between patches')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_mean', type=float, default=128.0)
    parser.add_argument('--image_stddev', type=float, default=33.0)
    parser.add_argument('--var_threshold', type=float, default=0.0,
                       help='Variance threshold (not used in PyTorch version)')

    return parser.parse_args()

def main():
    args = parse_args()
    model_args = json.loads(args.model_args)
    args.fov_size = tuple([int(i) for i in model_args['fov_size']])

    if 'label_size' in model_args:
        args.label_size = tuple([int(i) for i in model_args['label_size']])
    else:
        args.label_size = args.fov_size

    args.num_classes = int(model_args['num_classes'])

    inference = PyTorchInference(args)
    inference.run_inference()

if __name__ == '__main__':
    main()

