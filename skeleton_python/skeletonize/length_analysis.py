"""Skeleton length analysis: main trunk detection and branch measurement.

Coordinate Convention:
    Kimimaro outputs SWC files with coordinates in Z, Y, X order, which differs
    from the standard SWC format (X, Y, Z). This module preserves the Kimimaro
    convention for consistency with numpy array indexing (array[z, y, x]).

    All coordinate outputs (JSON, TIF) use Z, Y, X order.
"""
import os
import sys
import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import tifffile

from pipeline.utils import (
    auto_detect_subdir,
    get_output_dir,
    load_config,
    setup_logging,
    SKELETON_SCHEMA,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = BASE_DIR / 'pipeline' / 'skeleton_config.yaml'

# Visualization colors (R, G, B)
COLOR_TRUNK = (255, 255, 255)       # white - main trunk
COLOR_BRANCH_POINT = (255, 255, 0)  # yellow - branch point markers
COLOR_MAX_PATH = (0, 255, 0)        # bright green - max-length path

# Colors for other branch nodes (cycling)
BRANCH_COLORS = [
    # (0, 0, 255),      # blue
    # (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    # (255, 128, 0),    # orange
    # (128, 0, 255),    # purple
]


def parse_swc(swc_path: str) -> Tuple[Dict[int, dict], Dict[int, List[Tuple[int, float]]]]:
    """Parse SWC file and build graph structure.

    Args:
        swc_path: Path to SWC file.

    Returns:
        nodes: Dict mapping node ID to {z, y, x, radius, parent}.
               Coordinates are in Kimimaro SWC order (Z, Y, X).
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Note:
        Kimimaro outputs SWC with coordinates in Z, Y, X order, which differs
        from the standard SWC format (X, Y, Z). This function preserves the
        Kimimaro order for consistency with numpy array indexing [z, y, x].
    """
    nodes = {}
    adj = {}

    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            node_id = int(parts[0])
            # Kimimaro SWC format: ID TYPE Z Y X RADIUS PARENT
            # (differs from standard SWC which uses X Y Z)
            nodes[node_id] = {
                'z': float(parts[2]),
                'y': float(parts[3]),
                'x': float(parts[4]),
                'radius': float(parts[5]),
                'parent': int(parts[6]),
            }
            adj[node_id] = []

    # Build adjacency list with edge weights (Euclidean distance)
    for node_id, node in nodes.items():
        parent_id = node['parent']
        if parent_id != -1 and parent_id in nodes:
            parent = nodes[parent_id]
            dist = np.sqrt(
                (node['x'] - parent['x'])**2 +
                (node['y'] - parent['y'])**2 +
                (node['z'] - parent['z'])**2
            )
            adj[node_id].append((parent_id, dist))
            adj[parent_id].append((node_id, dist))

    return nodes, adj


def _bfs_farthest(start: int, adj: Dict[int, List[Tuple[int, float]]],
                  exclude: Set[int] = None) -> Tuple[int, float, Dict[int, int]]:
    """BFS to find farthest node from start.

    Args:
        start: Starting node ID.
        adj: Adjacency list.
        exclude: Set of node IDs to exclude from traversal.

    Returns:
        farthest_node: ID of the farthest node.
        max_dist: Distance to the farthest node.
        parent_map: Dict mapping each node to its parent in BFS tree.
    """
    if exclude is None:
        exclude = set()

    dist = {start: 0.0}
    parent_map = {start: -1}
    queue = deque([start])
    farthest_node = start
    max_dist = 0.0

    while queue:
        curr = queue.popleft()
        for neighbor, edge_dist in adj.get(curr, []):
            if neighbor in exclude or neighbor in dist:
                continue
            new_dist = dist[curr] + edge_dist
            dist[neighbor] = new_dist
            parent_map[neighbor] = curr
            queue.append(neighbor)

            if new_dist > max_dist:
                max_dist = new_dist
                farthest_node = neighbor

    return farthest_node, max_dist, parent_map


def find_main_trunk(nodes: Dict[int, dict],
                    adj: Dict[int, List[Tuple[int, float]]]) -> Tuple[List[int], float]:
    """Find main trunk using tree diameter algorithm (two BFS).

    Args:
        nodes: Node dictionary from parse_swc.
        adj: Adjacency list from parse_swc.

    Returns:
        trunk_path: List of node IDs forming the main trunk.
        trunk_length: Total length of the main trunk.
    """
    if not nodes:
        return [], 0.0

    # First BFS: from arbitrary node to find endpoint A
    start = next(iter(nodes.keys()))
    endpoint_a, _, _ = _bfs_farthest(start, adj)

    # Second BFS: from A to find endpoint B (farthest from A)
    endpoint_b, trunk_length, parent_map = _bfs_farthest(endpoint_a, adj)

    # Reconstruct path from A to B
    trunk_path = []
    curr = endpoint_b
    while curr != -1:
        trunk_path.append(curr)
        curr = parent_map.get(curr, -1)
    trunk_path.reverse()

    return trunk_path, trunk_length


def find_branch_points(trunk_path: List[int],
                       adj: Dict[int, List[Tuple[int, float]]]) -> List[int]:
    """Find branch points on the main trunk (degree >= 3).

    Args:
        trunk_path: List of node IDs forming the main trunk.
        adj: Adjacency list.

    Returns:
        List of node IDs that are branch points on the trunk.
    """
    branch_points = []

    for node_id in trunk_path:
        degree = len(adj.get(node_id, []))
        if degree >= 3:
            branch_points.append(node_id)

    return branch_points


def calculate_branch_max_length(branch_point: int, trunk_set: Set[int],
                                adj: Dict[int, List[Tuple[int, float]]]) -> float:
    """Calculate maximum branch length from a branch point.

    Finds the longest path from the branch point to any endpoint
    in the subtree (excluding main trunk nodes).

    Args:
        branch_point: Node ID of the branch point.
        trunk_set: Set of node IDs in the main trunk.
        adj: Adjacency list.

    Returns:
        Maximum branch length (distance to farthest endpoint).
    """
    # Find neighbors not on trunk (branch starting points)
    branch_starts = []
    for neighbor, dist in adj.get(branch_point, []):
        if neighbor not in trunk_set:
            branch_starts.append((neighbor, dist))

    if not branch_starts:
        return 0.0

    max_length = 0.0
    for start_node, start_dist in branch_starts:
        # BFS from this branch start, excluding trunk
        _, subtree_max, _ = _bfs_farthest(start_node, adj, exclude=trunk_set)
        total_length = start_dist + subtree_max
        if total_length > max_length:
            max_length = total_length

    return max_length


def find_max_length_path(branch_point: int, trunk_set: Set[int],
                         adj: Dict[int, List[Tuple[int, float]]]) -> Set[int]:
    """Find the nodes in the max-length path from a branch point.

    Args:
        branch_point: Node ID of the branch point.
        trunk_set: Set of node IDs in the main trunk.
        adj: Adjacency list.

    Returns:
        Set of node IDs in the max-length path (excluding branch_point itself).
    """
    # Find neighbors not on trunk (branch starting points)
    branch_starts = []
    for neighbor, dist in adj.get(branch_point, []):
        if neighbor not in trunk_set:
            branch_starts.append((neighbor, dist))

    if not branch_starts:
        return set()

    max_length = 0.0
    best_path = set()

    for start_node, start_dist in branch_starts:
        # BFS from this branch start, excluding trunk
        farthest, subtree_max, parent_map = _bfs_farthest(
            start_node, adj, exclude=trunk_set
        )
        total_length = start_dist + subtree_max

        if total_length > max_length:
            max_length = total_length
            # Reconstruct path from start_node to farthest
            path = set()
            curr = farthest
            while curr != -1:
                path.add(curr)
                curr = parent_map.get(curr, -1)
            best_path = path

    return best_path


def calculate_position_on_trunk(branch_point: int, trunk_path: List[int],
                                adj: Dict[int, List[Tuple[int, float]]]) -> float:
    """Calculate position of branch point on trunk (distance from start).

    Args:
        branch_point: Node ID of the branch point.
        trunk_path: List of node IDs forming the main trunk.
        adj: Adjacency list.

    Returns:
        Distance from trunk start to the branch point.
    """
    position = 0.0
    for i, node_id in enumerate(trunk_path):
        if node_id == branch_point:
            break
        if i < len(trunk_path) - 1:
            next_node = trunk_path[i + 1]
            # Find edge distance
            for neighbor, dist in adj.get(node_id, []):
                if neighbor == next_node:
                    position += dist
                    break
    return position


def calculate_total_length(adj: Dict[int, List[Tuple[int, float]]]) -> float:
    """Calculate total skeleton length (sum of all edges / 2).

    Args:
        adj: Adjacency list.

    Returns:
        Total skeleton length.
    """
    total = 0.0
    for neighbors in adj.values():
        for _, dist in neighbors:
            total += dist
    return total / 2  # Each edge counted twice


def analyze_skeleton(swc_path: str) -> dict:
    """Analyze skeleton structure from SWC file.

    Args:
        swc_path: Path to SWC file.

    Returns:
        Analysis result dictionary with summary, main_trunk, and branches.
    """
    nodes, adj = parse_swc(swc_path)

    if not nodes:
        return {
            'summary': {
                'total_nodes': 0,
                'total_length': 0.0,
                'main_trunk_length': 0.0,
                'num_branch_points': 0,
            },
            'main_trunk': {'start': [], 'end': [], 'length': 0.0},
            'branches': [],
        }

    trunk_path, trunk_length = find_main_trunk(nodes, adj)
    trunk_set = set(trunk_path)
    branch_point_ids = find_branch_points(trunk_path, adj)
    total_length = calculate_total_length(adj)

    # Get trunk endpoints (coordinates in Z, Y, X order - Kimimaro convention)
    start_node = nodes[trunk_path[0]] if trunk_path else {}
    end_node = nodes[trunk_path[-1]] if trunk_path else {}
    start_coords = [start_node.get('z', 0), start_node.get('y', 0), start_node.get('x', 0)]
    end_coords = [end_node.get('z', 0), end_node.get('y', 0), end_node.get('x', 0)]

    # Analyze each branch point
    branches = []
    for idx, bp_id in enumerate(branch_point_ids, start=1):
        bp_node = nodes[bp_id]
        position = calculate_position_on_trunk(bp_id, trunk_path, adj)
        max_length = calculate_branch_max_length(bp_id, trunk_set, adj)

        branches.append({
            'id': idx,
            # Coordinates in Z, Y, X order (Kimimaro convention)
            'branch_point': [bp_node['z'], bp_node['y'], bp_node['x']],
            'position_on_trunk': round(position, 2),
            'max_length': round(max_length, 2),
        })

    return {
        'coordinate_order': 'ZYX',
        'summary': {
            'total_nodes': len(nodes),
            'total_length': round(total_length, 2),
            'main_trunk_length': round(trunk_length, 2),
            'num_branch_points': len(branch_point_ids),
        },
        'main_trunk': {
            'start': [round(c, 2) for c in start_coords],
            'end': [round(c, 2) for c in end_coords],
            'length': round(trunk_length, 2),
        },
        'branches': branches,
    }


def generate_labeled_tif(swc_path: str, output_path: str,
                         shape: Tuple[int, int, int],
                         branch_point_radius: int = 2) -> None:
    """Generate RGB TIF visualization of skeleton analysis.

    Args:
        swc_path: Path to SWC file.
        output_path: Output TIF file path.
        shape: Image shape as (Z, Y, X) - numpy array order.
        branch_point_radius: Radius for branch point markers.
    """
    nodes, adj = parse_swc(swc_path)

    if not nodes:
        logger.warning("No nodes in skeleton, skipping TIF generation")
        return

    trunk_path, _ = find_main_trunk(nodes, adj)
    trunk_set = set(trunk_path)
    branch_point_ids = set(find_branch_points(trunk_path, adj))

    # Collect all max-length path nodes for all branch points
    max_path_nodes = set()
    for bp_id in branch_point_ids:
        path_nodes = find_max_length_path(bp_id, trunk_set, adj)
        max_path_nodes.update(path_nodes)

    # Create RGB image (Z, Y, X, 3)
    z_dim, y_dim, x_dim = shape
    rgb_image = np.zeros((z_dim, y_dim, x_dim, 3), dtype=np.uint8)

    # Assign branch IDs to non-trunk, non-max-path nodes
    branch_assignments = {}
    branch_counter = 0
    for bp_id in branch_point_ids:
        for neighbor, _ in adj.get(bp_id, []):
            if neighbor not in trunk_set and neighbor not in branch_assignments:
                # BFS to assign all nodes in this subtree
                visited = {neighbor}
                queue = deque([neighbor])
                while queue:
                    curr = queue.popleft()
                    branch_assignments[curr] = branch_counter
                    for next_node, _ in adj.get(curr, []):
                        if next_node not in trunk_set and next_node not in visited:
                            visited.add(next_node)
                            queue.append(next_node)
                branch_counter += 1

    # Draw skeleton points
    for node_id, node in nodes.items():
        # Kimimaro SWC coords are already in Z, Y, X order (matches numpy indexing)
        z = int(round(node['z']))
        y = int(round(node['y']))
        x = int(round(node['x']))

        if not (0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim):
            continue

        if node_id in branch_point_ids:
            # Branch point: yellow with radius
            for dz in range(-branch_point_radius, branch_point_radius + 1):
                for dy in range(-branch_point_radius, branch_point_radius + 1):
                    for dx in range(-branch_point_radius, branch_point_radius + 1):
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < z_dim and 0 <= ny < y_dim and 0 <= nx < x_dim:
                            rgb_image[nz, ny, nx] = COLOR_BRANCH_POINT
        elif node_id in trunk_set:
            # Main trunk: white
            rgb_image[z, y, x] = COLOR_TRUNK
        elif node_id in max_path_nodes:
            # Max-length path: bright green
            rgb_image[z, y, x] = COLOR_MAX_PATH
        elif node_id in branch_assignments:
            # Other branch nodes: cycling colors
            color_idx = branch_assignments[node_id] % len(BRANCH_COLORS)
            rgb_image[z, y, x] = BRANCH_COLORS[color_idx]
        else:
            # Unexpected node: red (for debugging)
            rgb_image[z, y, x] = (255, 0, 0)

    # Save as TIF (Z, Y, X, C order)
    # Note: For ImageJ compatibility with RGB, save without imagej=True
    # and let ImageJ auto-detect the format
    tifffile.imwrite(output_path, rgb_image, photometric='rgb')


def _find_cleaned_tif(swc_path: str, cleaned_dir: str = None) -> Optional[str]:
    """Find corresponding cleaned TIF file for an SWC file.

    Args:
        swc_path: Path to SWC file.
        cleaned_dir: Directory containing cleaned TIF files.

    Returns:
        Path to cleaned TIF file, or None if not found.
    """
    if cleaned_dir is None:
        # Try to find 03_cleaned directory from SWC path
        swc_dir = Path(swc_path).parent
        parent_dir = swc_dir.parent
        cleaned_dir = parent_dir / '03_cleaned'
        if not cleaned_dir.exists():
            return None
        cleaned_dir = str(cleaned_dir)

    # SWC: xxx_otsu_cleaned_label_1.swc -> TIF: xxx_otsu_cleaned.tif
    swc_stem = Path(swc_path).stem
    # Remove _label_N suffix
    if '_label_' in swc_stem:
        tif_stem = swc_stem.rsplit('_label_', 1)[0]
    else:
        tif_stem = swc_stem

    tif_path = os.path.join(cleaned_dir, f"{tif_stem}.tif")
    if os.path.exists(tif_path):
        return tif_path

    return None


def process_single_file(input_path: str, output_dir: str,
                        progress: str = "", cleaned_dir: str = None,
                        output_json: bool = True,
                        output_labeled_tif: bool = True,
                        branch_point_radius: int = 2) -> str:
    """Process single SWC file for length analysis.

    Args:
        input_path: Path to SWC file.
        output_dir: Output directory.
        progress: Progress indicator string.
        cleaned_dir: Directory containing cleaned TIF files (for shape).
        output_json: Whether to output JSON file.
        output_labeled_tif: Whether to output labeled TIF.
        branch_point_radius: Radius for branch point markers.

    Returns:
        Output directory path.
    """
    filename = os.path.basename(input_path)
    progress_prefix = f"[{progress}] " if progress else ""
    logger.info(f"{progress_prefix}Processing: {filename}")

    result = analyze_skeleton(input_path)

    os.makedirs(output_dir, exist_ok=True)
    input_stem = Path(input_path).stem

    # Save JSON
    if output_json:
        json_path = os.path.join(output_dir, f"{input_stem}_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.debug(f"  Saved JSON: {json_path}")

        # Save summary text file
        summary_path = os.path.join(output_dir, f"{input_stem}_summary.txt")
        trunk = result['main_trunk']
        with open(summary_path, 'w') as f:
            f.write(f"Trunk: {trunk['start']} -> {trunk['end']}, length={trunk['length']}\n\n")
            f.write("ID  Z      Y      X      MaxLen\n")
            for b in result['branches']:
                bp = b['branch_point']
                f.write(f"{b['id']:<3} {bp[0]:<6.0f} {bp[1]:<6.0f} {bp[2]:<6.0f} {b['max_length']}\n")
        logger.debug(f"  Saved summary: {summary_path}")

    # Save labeled TIF
    if output_labeled_tif:
        cleaned_tif = _find_cleaned_tif(input_path, cleaned_dir)
        if cleaned_tif:
            # Read shape from cleaned TIF (Z, Y, X)
            with tifffile.TiffFile(cleaned_tif) as tif:
                shape = tif.pages[0].shape
                if len(tif.pages) > 1:
                    shape = (len(tif.pages),) + shape
                elif tif.series[0].shape[0] > 1:
                    shape = tif.series[0].shape

            tif_path = os.path.join(output_dir, f"{input_stem}_labeled.tif")
            generate_labeled_tif(input_path, tif_path, shape,
                                 branch_point_radius)
            logger.debug(f"  Saved labeled TIF: {tif_path}")
        else:
            logger.warning("  Cleaned TIF not found, skipping labeled TIF output")

    # Log summary
    summary = result['summary']
    logger.info(f"  Trunk length: {summary['main_trunk_length']:.1f}, "
                f"Branch points: {summary['num_branch_points']}, "
                f"Total length: {summary['total_length']:.1f}")

    return output_dir


def process_directory(input_dir: str, output_dir: str = None,
                      cleaned_dir: str = None,
                      continue_on_error: bool = False,
                      **kwargs) -> None:
    """Process all SWC files in directory.

    Args:
        input_dir: Directory containing SWC files or timestamp directory.
        output_dir: Output directory (auto-detected if None).
        cleaned_dir: Directory containing cleaned TIF files.
        continue_on_error: If True, continue processing on file errors.
        **kwargs: Arguments passed to process_single_file.
    """
    input_dir_obj = Path(input_dir)
    if not input_dir_obj.is_absolute():
        input_dir_obj = BASE_DIR / input_dir
    input_dir = str(input_dir_obj.resolve())

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Auto-detect 04_skeleton subdirectory
    detected_dir = auto_detect_subdir(input_dir, '04_skeleton')
    if detected_dir != input_dir:
        input_dir = detected_dir
        logger.info(f"Auto-detected input directory: {input_dir}")

    # Auto-detect cleaned directory (for TIF shape)
    if cleaned_dir is None:
        parent_dir = Path(input_dir).parent
        cleaned_candidate = parent_dir / '03_cleaned'
        if cleaned_candidate.exists():
            cleaned_dir = str(cleaned_candidate)
            logger.debug(f"Auto-detected cleaned directory: {cleaned_dir}")

    # Auto-detect output directory
    if output_dir is None:
        output_dir = get_output_dir(input_dir, str(BASE_DIR / 'output'), '05_analysis')

    swc_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.swc')
    ])

    if not swc_files:
        raise ValueError(f"No SWC files found in directory: {input_dir}")

    logger.info(f"Found {len(swc_files)} SWC files in directory")
    logger.info(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    for idx, swc_file in enumerate(swc_files, start=1):
        try:
            process_single_file(
                swc_file, output_dir,
                progress=f"{idx}/{len(swc_files)}",
                cleaned_dir=cleaned_dir,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to process {swc_file}: {e}")
            failed_files.append((swc_file, str(e)))
            if not continue_on_error:
                raise

    successful = len(swc_files) - len(failed_files)
    logger.info(f"Completed processing {successful}/{len(swc_files)} files")

    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)}):")
        for path, error in failed_files:
            logger.warning(f"  {path}: {error}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze skeleton length: main trunk and branch measurements',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing SWC files or timestamp directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: auto-detect with timestamp)',
    )
    parser.add_argument(
        '--cleaned-dir',
        type=str,
        help='Directory containing cleaned TIF files (for image shape)',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (YAML)',
    )
    parser.add_argument(
        '--output-json',
        type=lambda x: x.lower() == 'true',
        default=None,
        help='Output JSON analysis file (true/false)',
    )
    parser.add_argument(
        '--output-labeled-tif',
        type=lambda x: x.lower() == 'true',
        default=None,
        help='Output labeled RGB TIF (true/false)',
    )
    parser.add_argument(
        '--branch-point-radius',
        type=int,
        help='Radius for branch point markers in TIF',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing other files if one fails',
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file',
    )

    args = parser.parse_args()

    setup_logging(log_file=args.log_file)

    # Load config file
    config = {}
    config_path = args.config or DEFAULT_CONFIG
    if os.path.exists(config_path):
        full_config = load_config(config_path, schema=SKELETON_SCHEMA)
        config = full_config.get('analysis', {})

    # CLI overrides
    if args.output_json is not None:
        config['output_json'] = args.output_json
    if args.output_labeled_tif is not None:
        config['output_labeled_tif'] = args.output_labeled_tif
    if args.branch_point_radius is not None:
        config['branch_point_radius'] = args.branch_point_radius

    # Defaults
    config.setdefault('output_json', True)
    config.setdefault('output_labeled_tif', True)
    config.setdefault('branch_point_radius', 2)

    try:
        process_directory(
            args.input_dir,
            output_dir=args.output_dir,
            cleaned_dir=args.cleaned_dir,
            continue_on_error=args.continue_on_error,
            **config
        )
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
