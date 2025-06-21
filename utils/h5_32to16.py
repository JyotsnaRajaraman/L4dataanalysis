import h5py
import numpy as np

def convert_hdf5_dtype(input_path, output_path, dataset_name):
    with h5py.File(input_path, 'r') as f_in:
        data = f_in[dataset_name][:]
        
        data_int16 = data.astype(np.int16)
        
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset(dataset_name, data=data_int16, compression="gzip")
    
    print(f"Converted '{dataset_name}' from int32 to int16 and saved to '{output_path}'.")


convert_hdf5_dtype('result.h5','result_int16.hdf5', 'segmentation')



