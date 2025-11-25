"""Run kimimaro skeletonization on binary mask."""

import sys
import os
import json
from datetime import datetime
import time
import numpy as np
from PIL import Image
import kimimaro


# Default configuration (fallback if config.json is missing parameters)
DEFAULT_CONFIG = {
    'teasar_params': {
        'scale': 1.5,
        'const': 300,
        'pdrf_scale': 100000,
        'pdrf_exponent': 4,
        'soma_detection_threshold': 1100,
        'soma_acceptance_threshold': 3500,
        'soma_invalidation_scale': 1,
        'soma_invalidation_const': 300,
        'max_paths': 300,
    },
    'dust_threshold': 1000,
    'anisotropy': [1, 1, 1],
    'fix_branching': True,
    'fix_borders': True,
    'progress': True,
    'parallel': 1,
    # Postprocessing parameters
    'postprocess_dust_threshold': 1000,
    'postprocess_tick_threshold': 3500,
    'keep_largest_component_only': False,
}


def load_config():
    """Load configuration from config.json."""
    config_file = 'config.json'

    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found")
        print("Please create config.json (use config.json.example as template)")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_file}: {e}")
        sys.exit(1)

    # Validate required parameters
    required = ['input_file', 'output_dir']
    missing = [key for key in required if key not in config]
    if missing:
        print(f"Error: Missing required parameters in config.json: {', '.join(missing)}")
        sys.exit(1)

    # Merge with defaults
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
        elif key == 'teasar_params' and isinstance(value, dict):
            # Merge teasar_params with defaults
            config[key] = {**value, **config[key]}

    return config


def load_tiff_stack(filepath):
    """Load TIFF stack into 3D numpy array."""
    img = Image.open(filepath)
    n_frames = getattr(img, 'n_frames', 1)

    frames = []
    for i in range(n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames)


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"


def main():
    # Start timing
    start_time = time.time()

    # Load configuration
    config = load_config()

    # Allow command line override
    if len(sys.argv) > 1:
        config['input_file'] = sys.argv[1]
    if len(sys.argv) > 2:
        config['output_dir'] = sys.argv[2]

    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%m%d%H%M')
    input_basename = os.path.splitext(os.path.basename(config['input_file']))[0]

    # Load mask
    try:
        labels = load_tiff_stack(config['input_file'])
        print(f"Input: {os.path.basename(config['input_file'])} "
              f"({labels.shape[0]}×{labels.shape[1]}×{labels.shape[2]})")
    except Exception as e:
        print(f"Error: Cannot load file: {e}")
        sys.exit(1)

    # Run skeletonization
    print("Processing...")
    skel_start = time.time()
    try:
        skels = kimimaro.skeletonize(
            labels,
            teasar_params=config['teasar_params'],
            dust_threshold=config['dust_threshold'],
            anisotropy=config['anisotropy'],
            fix_branching=config['fix_branching'],
            fix_borders=config['fix_borders'],
            progress=config['progress'],
            parallel=config['parallel'],
        )
    except Exception as e:
        import traceback
        print(f"Error: Skeletonization failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

    skel_time = time.time() - skel_start
    print(f"Skeletonization completed in {format_time(skel_time)}")

    # Save results
    if len(skels) == 0:
        print("No skeletons generated")
        return

    # Postprocessing: filter isolated fragments
    print("Postprocessing...")
    post_start = time.time()
    skels_filtered = {}
    for label_id, skel in skels.items():
        original_vertices = len(skel.vertices)

        # Apply kimimaro's postprocess function
        skel = kimimaro.postprocess(
            skel,
            dust_threshold=config.get('postprocess_dust_threshold', 1000),
            tick_threshold=config.get('postprocess_tick_threshold', 3500),
        )

        # Optionally keep only the largest connected component
        if config.get('keep_largest_component_only', False):
            components = skel.components()
            if len(components) > 1:
                print(f"  Label {label_id}: {len(components)} components found, keeping largest")
                skel = max(components, key=lambda c: c.cable_length())
            elif len(components) == 1:
                skel = components[0]

        filtered_vertices = len(skel.vertices)
        if filtered_vertices < original_vertices:
            print(f"  Label {label_id}: filtered {original_vertices} -> {filtered_vertices} vertices")

        skels_filtered[label_id] = skel

    skels = skels_filtered

    post_time = time.time() - post_start
    print(f"Postprocessing completed in {format_time(post_time)}")

    import tifffile

    os.makedirs(config['output_dir'], exist_ok=True)

    total_vertices = 0
    total_edges = 0

    for label_id, skel in skels.items():
        total_vertices += len(skel.vertices)
        total_edges += len(skel.edges)

        # Generate output filenames
        filename_base = f"{input_basename}_{timestamp}_label_{label_id}"
        swc_file = os.path.join(config['output_dir'], f'{filename_base}.swc')
        tif_file = os.path.join(config['output_dir'], f'{filename_base}.tif')

        # Save as SWC file
        try:
            swc_content = skel.to_swc(swc_file)
            with open(swc_file, 'w') as f:
                f.write(swc_content)
        except Exception as e:
            print(f"Warning: Failed to save SWC: {e}")

        # Create 3D image for visualization
        skeleton_img = np.zeros(labels.shape, dtype=np.uint8)

        # CRITICAL: Convert to voxel space
        # kimimaro returns vertices in physical space (scaled by anisotropy)
        skel_voxel = skel.voxel_space()
        vertices_int = skel_voxel.vertices.astype(int)

        # CRITICAL: Coordinate order
        # kimimaro vertices are in (z, y, x) order, matching numpy array indexing.
        # This was tested and verified in test_coordinate_order.py.
        #
        # DO NOT change to (x, y, z) - this will cause empty output!
        #
        # When numpy array is indexed as array[z, y, x]:
        #   - z corresponds to array.shape[0] (depth)
        #   - y corresponds to array.shape[1] (height)
        #   - x corresponds to array.shape[2] (width)
        #
        # kimimaro vertices[:,0] = z coordinate
        # kimimaro vertices[:,1] = y coordinate
        # kimimaro vertices[:,2] = x coordinate
        for v in vertices_int:
            z, y, x = v  # DO NOT change this order!
            if 0 <= z < labels.shape[0] and 0 <= y < labels.shape[1] and 0 <= x < labels.shape[2]:
                skeleton_img[z, y, x] = 255

        # Save as TIFF
        tifffile.imwrite(tif_file, skeleton_img, imagej=True)

    # Summary
    total_time = time.time() - start_time
    print(f"Output: {len(skels)} skeleton{'s' if len(skels) > 1 else ''}, "
          f"{total_vertices} vertices, {total_edges} edges")
    print(f"Saved to: {config['output_dir']}")
    print(f"Total time: {format_time(total_time)} (parallel={config['parallel']})")


if __name__ == '__main__':
    main()
