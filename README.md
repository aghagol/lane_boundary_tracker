# lane_boundary_tracker

Batch scripts for preprocessing, processing (tracking), visualization and evaluation for lane boundary tracking algorithm.

To run the complete process, run 

```bash
./scripts/runall --input [path to input data] --output [path to hold temporary files]
```

The configuration file `conf.json` is located at the repository root.

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
