import h5py
import wknml
import numpy as np
from collections import Counter

def load_segmentation_data(h5_file_path):
    """
    Load segmentation data from H5 file.
    
    Args:
        h5_file_path: Path to the H5 file containing segmentation data
        
    Returns:
        numpy.ndarray: 3D segmentation array
    """
    print(f"Loading segmentation data from {h5_file_path}...")
    
    with h5py.File(h5_file_path, 'r') as f:
        data = f['data'][:]
        print(f"  Data shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: {data.min()} - {data.max()}")
        print(f"  Unique segments: {len(np.unique(data))}")
        
    return data

def load_nml_thing(nml_file_path, thing_id):
    """
    Load a specific thing (tree) from NML file.
    
    Args:
        nml_file_path: Path to the NML file
        thing_id: ID of the thing/tree to load
        
    Returns:
        Tree object or None if not found
    """
    print(f"Loading thing ID {thing_id} from {nml_file_path}...")
    
    with open(nml_file_path, 'rb') as f:
        nml = wknml.parse_nml(f)
        
        # Find the specific thing
        target_thing = None
        for tree in nml.trees:
            if tree.id == thing_id:
                target_thing = tree
                break
        
        if target_thing:
            print(f"  Found thing ID {thing_id}: {getattr(target_thing, 'name', 'No name')}")
            print(f"  Number of nodes: {len(target_thing.nodes)}")
            return target_thing
        else:
            print(f"  Thing ID {thing_id} not found!")
            available_ids = sorted([tree.id for tree in nml.trees])
            print(f"  Available thing IDs: {available_ids[:20]}{'...' if len(available_ids) > 20 else ''}")
            return None

def get_thing_segment_ids(segmentation_data, thing, bbox_offset=None):
    """
    Find which segment IDs a thing (tree) is associated with.
    
    Args:
        segmentation_data: 3D segmentation array
        thing: Tree object from NML
        bbox_offset: Optional offset to apply to coordinates (for bounding box regions)
        
    Returns:
        Dictionary with segment analysis results
    """
    print(f"Analyzing segment IDs for thing {thing.id}...")
    
    segment_hits = Counter()  # segment_id -> count of nodes
    node_positions = []
    invalid_positions = []
    
    for i, node in enumerate(thing.nodes):
        # Get node position
        x, y, z = node.position
        
        # Apply offset if provided (for bounding box regions)
        if bbox_offset is not None:
            x -= bbox_offset[0]
            y -= bbox_offset[1] 
            z -= bbox_offset[2]
        
        # Convert to integer coordinates
        x, y, z = int(round(x)), int(round(y)), int(round(z))
        
        # Store position for analysis
        node_positions.append((x, y, z))
        
        # Check if position is within data bounds
        if (0 <= z < segmentation_data.shape[0] and
            0 <= y < segmentation_data.shape[1] and
            0 <= x < segmentation_data.shape[2]):
            
            # Get segment ID at this position
            segment_id = segmentation_data[z, y, x]
            segment_hits[segment_id] += 1
            
        else:
            invalid_positions.append((x, y, z, i))
            print(f"    Warning: Node {i} at position ({x}, {y}, {z}) is outside data bounds")
    
    # Analyze results
    total_nodes = len(thing.nodes)
    valid_nodes = total_nodes - len(invalid_positions)
    
    results = {
        'thing_id': thing.id,
        'thing_name': getattr(thing, 'name', 'No name'),
        'total_nodes': total_nodes,
        'valid_nodes': valid_nodes,
        'invalid_nodes': len(invalid_positions),
        'segment_hits': dict(segment_hits),
        'unique_segments': len(segment_hits),
        'node_positions': node_positions,
        'invalid_positions': invalid_positions
    }
    
    return results

def print_segment_analysis(results):
    """
    Print a detailed analysis of the segment results.
    
    Args:
        results: Dictionary from get_thing_segment_ids
    """
    print(f"\n=== Segment Analysis for Thing {results['thing_id']} ({results['thing_name']}) ===")
    print(f"Total nodes: {results['total_nodes']}")
    print(f"Valid nodes (within data bounds): {results['valid_nodes']}")
    print(f"Invalid nodes (outside bounds): {results['invalid_nodes']}")
    print(f"Unique segments hit: {results['unique_segments']}")
    
    if results['segment_hits']:
        print(f"\nSegment IDs and hit counts:")
        # Sort by hit count (descending)
        sorted_segments = sorted(results['segment_hits'].items(), key=lambda x: x[1], reverse=True)
        for segment_id, count in sorted_segments:
            percentage = (count / results['valid_nodes']) * 100 if results['valid_nodes'] > 0 else 0
            print(f"  Segment {segment_id}: {count} nodes ({percentage:.1f}%)")
        
        # Show most hit segment
        most_hit_segment = sorted_segments[0]
        print(f"\nMost hit segment: {most_hit_segment[0]} with {most_hit_segment[1]} nodes")
        
        # Check for background (segment ID 0)
        if 0 in results['segment_hits']:
            bg_hits = results['segment_hits'][0]
            print(f"Background hits (segment 0): {bg_hits} nodes")
    else:
        print("No valid segment hits found!")
    
    if results['invalid_positions']:
        print(f"\nInvalid positions (first 5):")
        for x, y, z, node_idx in results['invalid_positions'][:5]:
            print(f"  Node {node_idx}: ({x}, {y}, {z})")

def analyze_all_things(h5_file, nml_file, bbox_offset=None):
    """
    Step 1: Analyze all things in NML file and build thing_id -> segment_ids mapping.
    
    Args:
        h5_file: Path to H5 segmentation file
        nml_file: Path to NML file
        bbox_offset: Optional coordinate offset for bounding box regions
        
    Returns:
        tuple: (segmentation_data, thing_to_segments, segment_to_things)
    """
    print("=== Step 1: Analyzing All Things in NML File ===")
    
    # Load segmentation data
    segmentation_data = load_segmentation_data(h5_file)
    
    # Load all things from NML
    print(f"Loading all things from {nml_file}...")
    with open(nml_file, 'rb') as f:
        nml = wknml.parse_nml(f)
        print(f"Found {len(nml.trees)} things (trees) in NML file")
    
    # Dictionary to store thing_id -> set of segment_ids
    thing_to_segments = {}
    # Dictionary to store segment_id -> set of thing_ids (for overlap detection)
    segment_to_things = {}
    
    # Process each thing
    for i, tree in enumerate(nml.trees):
        thing_id = tree.id
        thing_name = getattr(tree, 'name', f'Thing_{thing_id}')
        
        print(f"Processing thing {thing_id} ({thing_name}) - {i+1}/{len(nml.trees)}")
        
        # Get segments for this thing
        results = get_thing_segment_ids(segmentation_data, tree, bbox_offset)
        
        if results['segment_hits']:
            # Store all segment IDs hit by this thing (excluding background segment 0)
            segment_ids = set(seg_id for seg_id in results['segment_hits'].keys() if seg_id != 0)
            thing_to_segments[thing_id] = segment_ids
            
            # Update reverse mapping for overlap detection
            for seg_id in segment_ids:
                if seg_id not in segment_to_things:
                    segment_to_things[seg_id] = set()
                segment_to_things[seg_id].add(thing_id)
            
            print(f"  Found {len(segment_ids)} segments: {sorted(list(segment_ids))}")
        else:
            print(f"  No valid segments found")
            thing_to_segments[thing_id] = set()
    
    print(f"\nStep 1 Complete:")
    print(f"  Total things processed: {len(thing_to_segments)}")
    print(f"  Things with segments: {sum(1 for segs in thing_to_segments.values() if segs)}")
    print(f"  Total unique segments found: {len(segment_to_things)}")
    
    return segmentation_data, thing_to_segments, segment_to_things

#work in progress
def check_overlaps_and_resolve(thing_to_segments, segment_to_things):
    """
    Step 2: Check for overlaps and resolve conflicts.
    
    Args:
        thing_to_segments: Dict mapping thing_id -> set of segment_ids
        segment_to_things: Dict mapping segment_id -> set of thing_ids
        
    Returns:
        tuple: (resolved_thing_to_segments, conflict_log)
    """
    print("\n=== Step 2: Checking for Overlaps and Resolving Conflicts ===")
    
    conflict_log = []
    resolved_thing_to_segments = thing_to_segments.copy()
    
    # Keep track of thing merges to handle cascading conflicts
    thing_merge_map = {}  # secondary_id -> primary_id
    
    # Find overlapping segments
    overlapping_segments = {seg_id: thing_ids for seg_id, thing_ids in segment_to_things.items() 
                           if len(thing_ids) > 1}
    
    if overlapping_segments:
        print(f"Found {len(overlapping_segments)} segments with overlaps:")
        
        for seg_id, thing_ids in overlapping_segments.items():
            # Resolve any thing IDs that have already been merged
            resolved_thing_ids = set()
            for tid in thing_ids:
                # Follow the merge chain to find the current primary ID
                current_id = tid
                while current_id in thing_merge_map:
                    current_id = thing_merge_map[current_id]
                resolved_thing_ids.add(current_id)
            
            # Skip if all things have been merged into the same primary
            if len(resolved_thing_ids) <= 1:
                continue
                
            thing_list = sorted(list(resolved_thing_ids))
            conflict_info = {
                'segment_id': seg_id,
                'conflicting_things': thing_list,
                'resolution': 'merge_things'
            }
            
            print(f"  Segment {seg_id} is claimed by things: {thing_list}")
            
            # Resolution strategy: Merge all conflicting things
            # Use the smallest thing_id as the primary ID
            primary_thing_id = min(thing_list)
            secondary_thing_ids = [tid for tid in thing_list if tid != primary_thing_id]
            
            print(f"    Resolution: Merging into thing {primary_thing_id}")
            
            # Merge all segments from secondary things into primary thing
            for secondary_id in secondary_thing_ids:
                if secondary_id in resolved_thing_to_segments:
                    # Add all segments from secondary thing to primary thing
                    if primary_thing_id in resolved_thing_to_segments:
                        resolved_thing_to_segments[primary_thing_id].update(
                            resolved_thing_to_segments[secondary_id]
                        )
                    else:
                        # Primary might have been deleted in a previous merge, create it
                        resolved_thing_to_segments[primary_thing_id] = resolved_thing_to_segments[secondary_id].copy()
                    
                    # Remove the secondary thing
                    del resolved_thing_to_segments[secondary_id]
                    
                    # Record the merge
                    thing_merge_map[secondary_id] = primary_thing_id
                    print(f"    Merged thing {secondary_id} into thing {primary_thing_id}")
            
            conflict_info['primary_thing_id'] = primary_thing_id
            conflict_info['merged_thing_ids'] = secondary_thing_ids
            conflict_log.append(conflict_info)
    else:
        print("No overlapping segments found - no conflicts to resolve!")
    
    print(f"\nStep 2 Complete:")
    print(f"  Conflicts resolved: {len(conflict_log)}")
    print(f"  Final number of things: {len(resolved_thing_to_segments)}")
    
    return resolved_thing_to_segments, conflict_log

def remap_segmentation(segmentation_data, thing_to_segments):
    """
    Step 3: Remap the segmentation data based on thing assignments.
    
    Args:
        segmentation_data: Original 3D segmentation array
        thing_to_segments: Dict mapping thing_id -> set of segment_ids
        
    Returns:
        numpy.ndarray: Remapped segmentation array
    """
    print("\n=== Step 3: Remapping Segmentation Data ===")
    
    # Create a copy for remapping
    remapped_data = segmentation_data.copy()
    
    # Create mapping from old_segment_id -> new_thing_id
    segment_to_thing_mapping = {}
    
    for thing_id, segment_ids in thing_to_segments.items():
        for seg_id in segment_ids:
            segment_to_thing_mapping[seg_id] = thing_id
    
    print(f"Remapping {len(segment_to_thing_mapping)} segments to {len(thing_to_segments)} things...")
    
    # TODO: Apply remapping
    segments_remapped = 0
    for old_seg_id, new_thing_id in segment_to_thing_mapping.items():
        mask = (segmentation_data == old_seg_id)
        voxel_count = np.sum(mask)
        if voxel_count > 0:
            remapped_data[mask] = new_thing_id
            segments_remapped += 1
            if segments_remapped <= 10:  # Show first 10 for progress
                print(f"  Remapped segment {old_seg_id} -> thing {new_thing_id} ({voxel_count} voxels)")
    
    print(f"Successfully remapped {segments_remapped} segments")
    
    original_unique = len(np.unique(segmentation_data))
    remapped_unique = len(np.unique(remapped_data))
    print(f"Original unique values: {original_unique}")
    print(f"Remapped unique values: {remapped_unique}")
    
    return remapped_data

def save_remapped_data(remapped_data, output_file, conflict_log):
    """
    Save the remapped segmentation data and conflict log.
    
    Args:
        remapped_data: Remapped 3D segmentation array
        output_file: Output H5 file path
        conflict_log: List of conflict resolution information
    """
    print(f"\nSaving remapped data to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Save remapped segmentation
        dataset = f.create_dataset(
            'remapped_segmentation',
            data=remapped_data,
            compression='gzip',
            compression_opts=6,
            chunks=True,
            shuffle=True
        )
        
        # Add metadata
        dataset.attrs['description'] = 'Segmentation remapped based on NML thing assignments'
        dataset.attrs['shape'] = remapped_data.shape
        dataset.attrs['dtype'] = str(remapped_data.dtype)
        dataset.attrs['unique_values'] = len(np.unique(remapped_data))
        dataset.attrs['min_value'] = int(remapped_data.min())
        dataset.attrs['max_value'] = int(remapped_data.max())
        
        # Save conflict log if any conflicts occurred
        if conflict_log:
            conflict_group = f.create_group('conflict_resolution')
            conflict_group.attrs['num_conflicts'] = len(conflict_log)
            
            for i, conflict in enumerate(conflict_log):
                conflict_subgroup = conflict_group.create_group(f'conflict_{i}')
                conflict_subgroup.attrs['segment_id'] = conflict['segment_id']
                conflict_subgroup.attrs['primary_thing_id'] = conflict['primary_thing_id']
                conflict_subgroup.create_dataset('conflicting_things', data=conflict['conflicting_things'])
                conflict_subgroup.create_dataset('merged_thing_ids', data=conflict['merged_thing_ids'])
    
    print(f"✓ Saved remapped data to {output_file}")

def complete_remapping_pipeline(h5_file, nml_file, output_file=None):
    """
    Complete pipeline for remapping segmentation based on NML things.
    
    Args:
        h5_file: Path to input H5 segmentation file
        nml_file: Path to NML file
        output_file: Path to output H5 file (optional)
        
    Returns:
        tuple: (remapped_data, conflict_log)
    """
    print("=== Complete NML-Based Segmentation Remapping Pipeline ===")
    
    # Calculate bounding box offset from filename
    bbox_min, bbox_max = get_bbox_from_filename(h5_file)
    bbox_offset = bbox_min
    
    # Step 1: Analyze all things and build mappings
    segmentation_data, thing_to_segments, segment_to_things = analyze_all_things(
        h5_file, nml_file, bbox_offset
    )
    
    # Step 2: Check for overlaps and resolve conflicts
    resolved_thing_to_segments, conflict_log = check_overlaps_and_resolve(
        thing_to_segments, segment_to_things
    )
    
    # Step 3: Remap segmentation data
    remapped_data = remap_segmentation(segmentation_data, resolved_thing_to_segments)
    
    # Save results if output file specified
    if output_file:
        save_remapped_data(remapped_data, output_file, conflict_log)
    
    return remapped_data, conflict_log

def get_bbox_from_filename(h5_file):
    """
    Calculate bounding box from H5 filename (e.g., x4y4z2.h5).
    
    Args:
        h5_file: Path to H5 file
        
    Returns:
        tuple: (bbox_min, bbox_max) as numpy arrays [x, y, z]
    """
    import os
    filename = os.path.basename(h5_file)
    
    # Parse filename like x4y4z2.h5
    # Extract x, y, z values
    parts = filename.replace('.h5', '').replace('x', '').replace('y', ' ').replace('z', ' ').split()
    x_idx, y_idx, z_idx = int(parts[0]), int(parts[1]), int(parts[2])
    
    # Calculate bounding box coordinates
    # x starts at 1024 * x_idx + 3, ends at that + 1024
    x_min = 1024 * x_idx + 3
    x_max = x_min + 1024
    
    y_min = 1024 * y_idx + 3  
    y_max = y_min + 1024
    
    z_min = 1024 * z_idx + 3
    z_max = z_min + 1024
    
    bbox_min = np.array([x_min, y_min, z_min])
    bbox_max = np.array([x_max, y_max, z_max])
    
    print(f"Calculated bounding box from filename {filename}:")
    print(f"  X: {x_min} - {x_max} (index {x_idx})")
    print(f"  Y: {y_min} - {y_max} (index {y_idx})")
    print(f"  Z: {z_min} - {z_max} (index {z_idx})")
    
    return bbox_min, bbox_max

if __name__ == "__main__":
    # Configuration
    H5_FILE = "../data/x4y4z2.h5"
    NML_FILE = "../data/region-1.nml"
    
    print("=== Analyzing Segment ID Overlaps ===")
    print("Finding all segment IDs that are claimed by multiple thing IDs...")
    
    # Calculate bounding box offset from filename
    bbox_min, bbox_max = get_bbox_from_filename(H5_FILE)
    bbox_offset = bbox_min
    
    # Step 1: Analyze all things and build mappings
    segmentation_data, thing_to_segments, segment_to_things = analyze_all_things(
        H5_FILE, NML_FILE, bbox_offset
    )
    
    # Find overlapping segments (segments claimed by multiple things)
    overlapping_segments = {seg_id: thing_ids for seg_id, thing_ids in segment_to_things.items() 
                           if len(thing_ids) > 1}
    
    print(f"\n=== OVERLAP ANALYSIS RESULTS ===")
    
    if overlapping_segments:
        print(f"Found {len(overlapping_segments)} segment IDs claimed by multiple thing IDs:")
        print()
        
        # Sort by segment ID for consistent output
        for seg_id in sorted(overlapping_segments.keys()):
            thing_ids = overlapping_segments[seg_id]
            thing_list = sorted(list(thing_ids))
            print(f"Segment ID {seg_id} is claimed by {len(thing_list)} things: {thing_list}")
        
        print(f"\nSUMMARY:")
        print(f"  Total segments with overlaps: {len(overlapping_segments)}")
        print(f"  Total segments analyzed: {len(segment_to_things)}")
        print(f"  Percentage with overlaps: {(len(overlapping_segments) / len(segment_to_things) * 100):.1f}%")
        
        # Show distribution of overlap counts
        overlap_counts = {}
        for thing_ids in overlapping_segments.values():
            count = len(thing_ids)
            overlap_counts[count] = overlap_counts.get(count, 0) + 1
        
        print(f"\nOverlap distribution:")
        for num_things in sorted(overlap_counts.keys()):
            num_segments = overlap_counts[num_things]
            print(f"  {num_segments} segments claimed by {num_things} things")
            
    else:
        print("No overlapping segments found!")
        print("All segment IDs are uniquely assigned to single thing IDs.")
        print(f"Total segments analyzed: {len(segment_to_things)}")
    
    print(f"\nAnalysis complete. Found overlaps for {len(overlapping_segments)} segment IDs.")
