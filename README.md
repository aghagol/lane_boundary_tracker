# lane_boundary_tracker

Batch scripts for preprocessing, processing (tracking), visualization and evaluation for lane boundary tracking algorithm.

To execute the complete process, run:

```bash
./scripts/runall \
  --input   path/to/data_dir \
  --drives  path/to/drive_names_txt \
  --poses   path/to/csv_poses_dir \
  --chunks  path/to/chunks_metadata_dir \
  --tagged  path/to/tagged_data_dir \
  --output  path/to/tmp_cache_dir
```

To just visualize (given that line-connection results have been produced and output), run:

```bash
./scripts/runall --output path/to/tmp_cache_dir
```

 - The configuration file `conf.json` is located at the repository root.
 - The text file `drive_names_txt` contains IDs of drives to be processed.
 - If the `--drives` option is not used, a `drive_list.txt` is generated in the output directory.
 - If the `--drives` input file does not point to an existing file, it is generated.
 - Note that we need access to drive metadata (e.g. pose and chunks metadata).

The `runall` scripts runs the following scripts:

## Preprocessing:

```bash
./scripts/preprocess
```

This script:
 - Breaks drive-based (pose-centric) JSON files to smaller surface-based JSON files
 - Extracts detection points and tracking ground-truth CSV files from JSON files
 - Optional: applies detection noise and drop

## (Lane boundary) Tracking:

```bash
./scripts/sort
```

This script:
 - Applied SORT tracking over detections and saves the results in CSV format

## Evaluation:

```bash
./scripts/evaluate
```

This script:
 - Converts tracking and ground-truth CSV files to SLOTH formatted JSON files
 - Applies PYMOT evaluation (multi-object tracking metrics) on JSON files and prints the output

## Visualization:

```bash
./scripts/visualize
```

This script:
 - Displays the tracking CSV output of tracking algorithm (e.g. SORT)
