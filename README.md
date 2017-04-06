# lane_boundary_tracker

Batch scripts for preprocessing, processing (tracking), visualization and evaluation for lane boundary tracking algorithm.

To run the complete process, run 

```sh
./scripts/runall
```

The `runall` scripts runs the following scripts except for visualization, which can be run separately through `./scripts/visualize`:

## Preprocessing:

```sh
./scripts/preprocess
```

This script:
 - Breaks drive-based (pose-centric) JSON files to smaller surface-based JSON files
 - Extracts detection points and tracking ground-truth CSV files from JSON files
 - Optional: applies detection noise and drop

## (Lane boundary) Tracking:

```sh
./scripts/sort
```

This script:
 - Applied SORT tracking over detections and saves the results in CSV format

## Evaluation:

```sh
./scripts/evaluate
```

This script:
 - Converts tracking and ground-truth CSV files to SLOTH formatted JSON files
 - Applies PYMOT evaluation (multi-object tracking metrics) on JSON files and prints the output

## Visualization:

```sh
./scripts/visualize
```

This script:
 - Displays the tracking CSV output of tracking algorithm (e.g. SORT)
