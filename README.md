# Skeleton Pipeline

A complete pipeline for 3D image skeletonization and analysis.

## Demo

### Skeleton Extraction
![Skeleton](skeleton_python/output/example_output_projections/sample_input_otsu_cleaned_label_1_projections.gif)

### Analysis - Labeled Structure
![Labeled](skeleton_python/output/example_output_projections/sample_input_otsu_cleaned_label_1_labeled_projections.gif)

### Analysis - Length Map
![Length](skeleton_python/output/example_output_projections/sample_input_otsu_cleaned_label_1_length_projections.gif)

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

## Sample Data

Download sample data from Google Drive:
- [Sample Data Folder](https://drive.google.com/drive/folders/1w4wIAczOLmvhfEuUUNyLAlcUoYi9cey5?usp=drive_link)

Place downloaded files in `skeleton_python/DATA/` directory.

## Installation

```bash
cd skeleton_python/skeleton_pipeline
pip install -r requirements.txt
```

## Usage

### Run full pipeline

```bash
cd skeleton_python/skeleton_pipeline
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

### Use custom config

```bash
python run.py --input ../DATA/filopodia/sample.tif --config config/config_filopodia.yaml
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

Configuration files are in `skeleton_pipeline/config/`:

- `config.yaml` - Default parameters (for general use)
- `config_filopodia.yaml` - Optimized for thin filopodia structures

Edit `config/config.yaml` to customize pipeline parameters:

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
├── config_used.yaml     # Copy of config used for this run
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
