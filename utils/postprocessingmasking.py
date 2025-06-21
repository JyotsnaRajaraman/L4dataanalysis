import h5py
import numpy as np
from scipy.ndimage import binary_fill_holes

with h5py.File("x1y5z1_predictions_only_binary_uint8.h5", "r") as f:
    mask = f["predictions_binary"][:]  # shape: (Z, Y, X)

# 2D fill, slice by slice
filled_mask = np.zeros_like(mask, dtype=bool)
for z in range(mask.shape[0]):
    filled_mask[z] = binary_fill_holes(mask[z].astype(bool))

with h5py.File("x1y5z1_predictions_2dfill_scipy.h5", "w") as f:
    f.create_dataset("filled", data=filled_mask.astype(np.uint8), compression="gzip")


import argparse
import h5py
import numpy as np
from scipy.ndimage import binary_fill_holes

#works better
def fill_2d_holes(input_path, output_path, input_key="data", output_key="data"):
    with h5py.File(input_path, "r") as f:
        mask = f[input_key][:]  # shape: (Z, Y, X)

    filled_mask = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        filled_mask[z] = binary_fill_holes(mask[z].astype(bool))

    with h5py.File(output_path, "w") as f:
        f.create_dataset(output_key, data=filled_mask.astype(np.uint8), compression="gzip")


# does not close all artifacts
# def fill_3d_holes(input_path, output_path, input_key="data", output_key="data"):
#     with h5py.File(input_path, "r") as f:
#         mask = f[input_key][:]  # shape: (Z, Y, X)

#     mask = mask.astype(bool)
#     filled_mask = binary_fill_holes(mask)

#     with h5py.File(output_path, "w") as f:
#         f.create_dataset(output_key, data=filled_mask.astype(np.uint8), compression="gzip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hole filling for binary mask")
    parser.add_argument("--input", required=True, help="input h5 file path")
    parser.add_argument("--output", required=True, help="output h5 file path")
    parser.add_argument("--input-key", default="data", help="key in input h5 (default: data)")
    parser.add_argument("--output-key", default="data", help="key to use in output h5 (default: data)")

    args = parser.parse_args()

    fill_2d_holes(args.input, args.output, args.input_key, args.output_key)

