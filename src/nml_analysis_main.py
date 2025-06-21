import wknml
import numpy as np
from load_h5_bbox import load_hdf5_with_bbox
from collections import defaultdict
import time
def load_nml_files():
    """Load the three specific NML files in the directory with their predefined bounding boxes."""
    
    # The three specific NML files
    nml_files = ["region-1.nml"]#, "region-2.nml", "region-3.nml"]
    
    # Predefined bounding boxes from MATLAB code
    # Format: [min_x, min_y, min_z, max_x, max_y, max_z]
    # MATLAB: pT.local(1).bboxSmall = [4133 3963 2253; 4578 4408 2432]'+1;
    # Note: +1 is already applied in these values (MATLAB to Python conversion)
    bounding_boxes = {
        "region-1.nml": np.array([[4134, 3964, 2254], [4579, 4409, 2433]])  # [min_xyz, max_xyz]
       # "region-2.nml": np.array([[4439, 1321, 894], [4884, 1766, 1073]]),
       # "region-3.nml": np.array([[1825, 6674, 1240], [2270, 7119, 1419]])
    }
    
    nmls = {}
    
    # Load each NML file
    for nml_path in nml_files:
        try:
            print(f"Loading {nml_path}...")
            
            with open(nml_path, 'rb') as f:
                nml = wknml.parse_nml(f)
            
            # Calculate bounding box from node positions
            voxel_coords = []
            for tree in nml.trees:
                for node in tree.nodes:
                    # Get original node position
                    pos_x, pos_y, pos_z = node.position
                    voxel_coords.append((pos_x, pos_y, pos_z))
            
            # Convert to numpy array and find min/max
            if voxel_coords:
                voxel_coords = np.array(voxel_coords)
                node_bbox_min = np.min(voxel_coords, axis=0).astype(int)
                node_bbox_max = np.max(voxel_coords, axis=0).astype(int)
            else:
                node_bbox_min = np.array([0, 0, 0])
                node_bbox_max = np.array([0, 0, 0])
            
            # Get the predefined bounding box
            predefined_bbox = bounding_boxes.get(nml_path)
            predefined_min = predefined_bbox[0]
            predefined_max = predefined_bbox[1]
            
            # Store NML object with filename as key
            nmls[nml_path] = {
                'nml': nml,
                'node_bbox': {
                    'min': node_bbox_min,  # [min_x, min_y, min_z]
                    'max': node_bbox_max   # [max_x, max_y, max_z]
                },
                'predefined_bbox': {
                    'min': predefined_min,
                    'max': predefined_max
                },
                # Calculate maximum covering bounding box
                'bbox': {
                    'min': np.minimum(node_bbox_min, predefined_min),
                    'max': np.maximum(node_bbox_max, predefined_max)
                }
            }
            
            # Print basic info
            print(f"  Trees: {len(nml.trees)}")
            print(f"  Total nodes: {sum(len(tree.nodes) for tree in nml.trees)}")
            print(f"  Node-based bounding box: Min XYZ: {node_bbox_min}, Max XYZ: {node_bbox_max}")
            print(f"  Node-based dimensions: {node_bbox_max - node_bbox_min + 1}")  # +1 because inclusive bounds
            print(f"  Predefined bounding box: Min XYZ: {predefined_min}, Max XYZ: {predefined_max}")
            print(f"  Predefined dimensions: {predefined_max - predefined_min + 1}")  # +1 because inclusive bounds
            
            # Get the maximum covering bounding box
            max_cover_min = nmls[nml_path]['bbox']['min']
            max_cover_max = nmls[nml_path]['bbox']['max']
            print(f"  Maximum covering bounding box: Min XYZ: {max_cover_min}, Max XYZ: {max_cover_max}")
            print(f"  Maximum covering dimensions: {max_cover_max - max_cover_min + 1}")  # +1 because inclusive bounds
        
        except FileNotFoundError:
            print(f"  Error: File {nml_path} not found")
        except Exception as e:
            print(f"  Error loading {nml_path}: {e}")
    
    return nmls

def load_data_for_nml(nml_info, base_dir='.', dataset_name='data'):
    """
    Load H5 data for the specified NML file using its bounding box.
    
    Args:
        nml_info: Dictionary with NML data and bounding box
        base_dir: Directory containing H5 files
        dataset_name: Name of the dataset inside each H5 file
        
    Returns:
        numpy.ndarray: 3D array containing the data for the NML bounding box
    """
    bbox_min = nml_info['bbox']['min']
    bbox_max = nml_info['bbox']['max']
    
    # Convert to load_h5_with_bbox format: (min_x, max_x, min_y, max_y, min_z, max_z)
    bbox_tuple = (
        int(bbox_min[0]), int(bbox_max[0]),
        int(bbox_min[1]), int(bbox_max[1]),
        int(bbox_min[2]), int(bbox_max[2])
    )
    
    # Get array dimensions
    width = bbox_max[0] - bbox_min[0] + 1
    height = bbox_max[1] - bbox_min[1] + 1
    depth = bbox_max[2] - bbox_min[2] + 1
    
    print(f"Loading data for bounding box: X[{bbox_min[0]}-{bbox_max[0]}], Y[{bbox_min[1]}-{bbox_max[1]}], Z[{bbox_min[2]}-{bbox_max[2]}]")
    print(f"Expected dimensions: {width}x{height}x{depth}")
    
    # Load data
    try:
        data = load_hdf5_with_bbox(bbox_tuple, base_dir, dataset_name)
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    print("Number of unique segments in bbox")
    print(np.unique(data))
def perform_segment_pickup(nml, data, bbox_min):
    """
    Identify segments in the data that are traversed by the NML skeleton.
    
    Args:
        nml: The NML object
        data: 3D array of segmentation data
        bbox_min: Minimum coordinates of the bounding box
    
    Returns:
        Dictionary mapping tree IDs to their segment information
    """
    print("Performing segment pickup...")
    start_time = time.time()
    counter =0

    # Track segments per tree
    tree_segments = {}
    
    # Keep all segments encountered
    all_segments = set()
    
    for tree in nml.trees:
        tree_id = tree.id
        segments = defaultdict(int)  # segment_id -> count
        
        # Track positions in data volume
        positions = []
        
        for node in tree.nodes:
            # Get original coordinates
            pos_x, pos_y, pos_z = node.position
            
            # Convert to voxel coordinates in the global space
            vox_x = int(pos_x)
            vox_y = int(pos_y)
            vox_z = int(pos_z)
            
            # Convert to local coordinates within the bounding box
            local_x = vox_x - bbox_min[0]
            local_y = vox_y - bbox_min[1]
            local_z = vox_z - bbox_min[2]
            
            # Store the position
            positions.append((local_x, local_y, local_z))
            
            # Check if position is within the data volume
            if (0 <= local_z < data.shape[0] and 
                0 <= local_y < data.shape[1] and 
                0 <= local_x < data.shape[2]):
                
                # Get segment ID at this position
                segment_id = data[local_z, local_y, local_x]
                
                # Increment count for this segment
                if segment_id > 0:  # Skip background
                    segments[segment_id] += 1
                    all_segments.add(segment_id)
            else:
                counter+=1
        # Store segments for this tree
        tree_segments[tree_id] = {
            'name': tree.name,
            'segments': dict(segments),
            'node_count': len(tree.nodes),
            'positions': positions
        }
    
    print(f"Segment pickup completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(all_segments)} unique segments across all trees")
    print("Skipped nodes")
    print(counter)
    return tree_segments, all_segments

if __name__ == "__main__":
    # Load the three NML files with their bounding boxes
    nmls = load_nml_files()
    
    # Process each NML file
    for nml_file, nml_info in nmls.items():
        print(f"\n=== Processing {nml_file} ===")
        
        # Load data for this NML file using the maximum covering bbox
        print("\nLoading segmentation data for maximum covering bbox:")
        data = load_data_for_nml(nml_info)
        
        if data is not None:
            # Store the data in the NML info dictionary
            nml_info['data'] = data
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Min value: {data.min()}, Max value: {data.max()}")
            
            # Count unique values (segments) in the full data
            all_unique_segments = np.unique(data)
            all_unique_segments = all_unique_segments[all_unique_segments > 0]  # Exclude background
            print(f"Number of unique segments in volume: {len(all_unique_segments)}")
            
            # Perform segment pickup
            print("\nPerforming segment pickup:")
            tree_segments, segments_picked_up = perform_segment_pickup(
                nml_info['nml'], 
                data, 
                nml_info['bbox']['min']
            )
            
            # Store segment info
            nml_info['tree_segments'] = tree_segments
            nml_info['segments_picked_up'] = segments_picked_up
            
            # Print summary of segments per tree
            print("\nSegments per tree:")
            for tree_id, tree_info in tree_segments.items():
                num_segments = len(tree_info['segments'])
                print(f"Tree {tree_id} ({tree_info['name']}): {num_segments} segments, {tree_info['node_count']} nodes")
                
                # Print top segments (by node count)
                if tree_info['segments']:
                    top_segments = sorted(tree_info['segments'].items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  Top segments: {', '.join([f'ID {s[0]} ({s[1]} nodes)' for s in top_segments])}")
            
            print(f"\nTotal unique segments across all trees: {len(segments_picked_up)}")
            
            # Now load data for the predefined bounding box specifically
            print("\nLoading predefined bounding box data for segment analysis:")
            
            # Temporarily modify the nml_info to use predefined bbox
            original_bbox = nml_info['bbox'].copy()
            nml_info['bbox'] = nml_info['predefined_bbox']
            
            # Load the data for the predefined bbox
            predefined_data = load_data_for_nml(nml_info)
            
            # Restore the original bbox
            nml_info['bbox'] = original_bbox
            
            if predefined_data is not None:
                # Count unique segments in predefined bbox
                predefined_segments = np.unique(predefined_data)
                predefined_segments = predefined_segments[predefined_segments > 0]  # Exclude background
                
                # Calculate segments not picked up by NML nodes
                segments_not_picked_up = set(predefined_segments) - set(segments_picked_up)
                
                # Store the statistics
                nml_info['predefined_segments'] = set(predefined_segments)
                nml_info['segments_not_picked_up'] = segments_not_picked_up
                
                # Print the final statistics
                print("\n--- FINAL STATISTICS ---")
                print(f"Total segments in predefined bounding box: {len(predefined_segments)}")
                print(f"Segments picked up by NML nodes: {len(segments_picked_up)}")
                print(f"Segments in predefined bbox NOT picked up by NML nodes: {len(segments_not_picked_up)}")
                print(f"Percentage of segments not picked up: {len(segments_not_picked_up) / len(predefined_segments) * 100:.2f}%")
                
                if len(segments_not_picked_up) > 0:
                    print(f"First 10 segments not picked up: {sorted(list(segments_not_picked_up))[:10]}")

