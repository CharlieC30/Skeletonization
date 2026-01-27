# Skeleton Pipeline

A complete pipeline for 3D image skeletonization and analysis.

## Overview

This pipeline extracts skeleton structures from 3D TIF images and analyzes their morphology (main trunk, branch points, lengths).

## Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | Format conversion | Normalize TIF and convert to uint8 |
| 2 | Otsu thresholding | Binarize using Otsu's method |
| 3 | Mask cleaning | Morphological operations (opening, closing, fill holes) |
| 4 | Skeletonization | Extract skeleton using Kimimaro TEASAR algorithm |
| 5 | Length analysis | Analyze main trunk and branches |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run full pipeline

```bash
cd skeleton_pipeline
python run.py --input ../DATA/input.tif
```

### Specify output directory

```bash
python run.py --input ../DATA/input.tif --output /path/to/output
```

### Run only specific step

```bash
python run.py --input ../DATA/input.tif --step 3
```

### Run step range

```bash
python run.py --input ../DATA/input.tif --from 2 --to 4
```

### Override parameters

```bash
python run.py --input ../DATA/input.tif --dust-threshold 1000 --opening-radius 2
```

### Show help

```bash
python run.py --help
```

## Configuration

Edit `config.yaml` to customize pipeline parameters:

```yaml
# Output settings
output:
  base_dir: "../output"
  use_timestamp: true

# Step 1: Format conversion
format_conversion:
  normalize_method: minmax

# Step 3: Mask cleaning
clean_masks:
  opening_radius: 1
  closing_radius: 2
  min_size: 64

# Step 4: Skeletonization
skeletonization:
  dust_threshold: 500
  parallel: 1

# Step 5: Analysis
analysis:
  output_json: true
  output_labeled_tif: true
```

## Output Structure

```
output/YYYYMMDD_HHMMSS/
├── 01_format/           # Normalized TIF files
├── 02_otsu/             # Binary masks (*_otsu.tif)
├── 03_cleaned/          # Cleaned masks (*_otsu_cleaned.tif)
├── 04_skeleton/         # SWC files + skeleton TIFs
└── 05_analysis/         # JSON analysis + labeled TIFs
```

## Coordinate Convention

This pipeline uses **Z, Y, X** coordinate order (numpy array indexing convention), which differs from standard SWC format (X, Y, Z). All outputs include coordinate order information.

## Input Formats

- Single 3D TIF file
- Directory of 3D TIF files
- Directory of 2D TIF sequence (auto-stacked)

## Dependencies

- numpy
- tifffile
- scipy
- scikit-image
- kimimaro
- natsort
- pyyaml
