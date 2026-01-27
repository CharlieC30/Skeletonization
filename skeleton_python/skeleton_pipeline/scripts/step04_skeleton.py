"""Step 04: 3D skeletonization using Kimimaro algorithm."""
import os
import logging
from pathlib import Path

import numpy as np
import tifffile
import kimimaro
from natsort import natsorted


STEP_NAME = "04_skeleton"
STEP_DESCRIPTION = "Skeletonization"
REQUIRES = "03_cleaned"


def skeletonize_mask(
    image: np.ndarray,
    teasar_params: dict = None,
    dust_threshold: int = 500,
    anisotropy: tuple = (1, 1, 1),
    fix_branching: bool = True,
    fix_borders: bool = True,
    progress: bool = True,
    parallel: int = 1,
    postprocess_dust_threshold: int = 1000,
    postprocess_tick_threshold: int = 0,
    keep_largest_component_only: bool = True,
    logger: logging.Logger = None,
) -> dict:
    """Skeletonize binary mask using Kimimaro.

    Args:
        image: 3D binary image.
        teasar_params: TEASAR algorithm parameters.
        dust_threshold: Remove components smaller than this before skeletonization.
        anisotropy: Voxel anisotropy (Z, Y, X).
        fix_branching: Fix branching structures.
        fix_borders: Fix border issues.
        progress: Show progress bar.
        parallel: Number of parallel processes.
        postprocess_dust_threshold: Remove skeleton fragments shorter than this.
        postprocess_tick_threshold: Remove terminal branches shorter than this.
        keep_largest_component_only: Keep only largest component.
        logger: Logger instance.

    Returns:
        Dictionary of skeleton objects keyed by label ID.
    """
    if teasar_params is None:
        teasar_params = {
            'scale': 0.3,
            'const': 50,
            'pdrf_scale': 100000,
            'pdrf_exponent': 4,
            'soma_detection_threshold': 999999,
            'soma_acceptance_threshold': 999999,
            'soma_invalidation_scale': 2,
            'soma_invalidation_const': 300,
            'max_paths': None,
        }

    # Convert to binary labels
    labels = (image > 0).astype(np.uint8)

    # Run Kimimaro skeletonization
    if logger:
        logger.debug("Running Kimimaro skeletonization")

    skels = kimimaro.skeletonize(
        labels,
        teasar_params=teasar_params,
        dust_threshold=dust_threshold,
        anisotropy=anisotropy,
        fix_branching=fix_branching,
        fix_borders=fix_borders,
        progress=progress,
        parallel=parallel,
    )

    if len(skels) == 0:
        return {}

    # Postprocessing
    if logger:
        logger.debug("Postprocessing skeletons")

    skels_filtered = {}
    for label_id, skel in skels.items():
        original_vertices = len(skel.vertices)

        skel = kimimaro.postprocess(
            skel,
            dust_threshold=postprocess_dust_threshold,
            tick_threshold=postprocess_tick_threshold,
        )

        if keep_largest_component_only:
            components = skel.components()
            if len(components) > 1:
                if logger:
                    logger.debug(f"  Label {label_id}: {len(components)} components, keeping largest")
                skel = max(components, key=lambda c: c.cable_length())
            elif len(components) == 1:
                skel = components[0]

        filtered_vertices = len(skel.vertices)
        if logger and filtered_vertices < original_vertices:
            logger.debug(f"  Label {label_id}: filtered {original_vertices} -> {filtered_vertices} vertices")

        skels_filtered[label_id] = skel

    return skels_filtered


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run Step 04: Skeletonization.

    Args:
        input_path: Path to input (previous step output directory).
        output_dir: Base output directory.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Output directory path for this step.
    """
    # Input is previous step's output directory
    input_dir = Path(input_path)
    if not input_dir.exists():
        input_dir = Path(output_dir) / "03_cleaned"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = Path(output_dir) / STEP_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    # Get config parameters
    skel_config = config.get('skeletonization', {})
    teasar_params = skel_config.get('teasar_params', {})
    dust_threshold = skel_config.get('dust_threshold', 500)
    anisotropy = skel_config.get('anisotropy', [1, 1, 1])
    if isinstance(anisotropy, list):
        anisotropy = tuple(anisotropy)
    fix_branching = skel_config.get('fix_branching', True)
    fix_borders = skel_config.get('fix_borders', True)
    progress = skel_config.get('progress', True)
    parallel = skel_config.get('parallel', 1)
    postprocess_dust_threshold = skel_config.get('postprocess_dust_threshold', 1000)
    postprocess_tick_threshold = skel_config.get('postprocess_tick_threshold', 0)
    keep_largest_component_only = skel_config.get('keep_largest_component_only', True)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Parameters: dust_threshold={dust_threshold}, parallel={parallel}")

    # Find *_cleaned.tif files
    tif_files = natsorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff')
        and '_cleaned' in f.name
    ])

    if not tif_files:
        raise ValueError(f"No *_cleaned.tif files found in: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files")

    for idx, tif_file in enumerate(tif_files, start=1):
        logger.info(f"[{idx}/{len(tif_files)}] Processing: {tif_file.name}")

        image = tifffile.imread(str(tif_file))
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image.ndim}D")

        logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

        skels = skeletonize_mask(
            image,
            teasar_params=teasar_params,
            dust_threshold=dust_threshold,
            anisotropy=anisotropy,
            fix_branching=fix_branching,
            fix_borders=fix_borders,
            progress=progress,
            parallel=parallel,
            postprocess_dust_threshold=postprocess_dust_threshold,
            postprocess_tick_threshold=postprocess_tick_threshold,
            keep_largest_component_only=keep_largest_component_only,
            logger=logger,
        )

        if len(skels) == 0:
            logger.warning("No skeletons generated")
            continue

        input_name = tif_file.stem
        total_vertices = 0
        total_edges = 0

        for label_id, skel in skels.items():
            total_vertices += len(skel.vertices)
            total_edges += len(skel.edges)

            # Save SWC file
            swc_file = output_path / f'{input_name}_label_{label_id}.swc'
            try:
                swc_content = skel.to_swc()
                with open(swc_file, 'w') as f:
                    f.write(swc_content)
            except Exception as e:
                logger.warning(f"Failed to save SWC: {e}")

            # Save TIF visualization
            tif_out_file = output_path / f'{input_name}_label_{label_id}.tif'
            skeleton_img = np.zeros(image.shape, dtype=np.uint8)

            skel_voxel = skel.voxel_space()
            vertices_int = skel_voxel.vertices.astype(int)

            for v in vertices_int:
                z, y, x = v
                if 0 <= z < image.shape[0] and 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                    skeleton_img[z, y, x] = 255

            tifffile.imwrite(str(tif_out_file), skeleton_img, imagej=True, metadata={'axes': 'ZYX'})

        logger.debug(f"  Generated {len(skels)} skeleton(s), {total_vertices} vertices, {total_edges} edges")

    return str(output_path)
