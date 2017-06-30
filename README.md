# lane_boundary_tracker

Batch scripts for preprocessing, processing (tracking), visualization and evaluation for lane boundary tracking algorithm.

To execute the complete process, run:

```bash
./scripts/runall \
  --input   path/to/CNN_predictions \
  --fuses   path/to/fuse_files \
  --tagged  path/to/tagged_fuse_files \
  --drives  list_of_drives_txt_file \
  --poses   path/to/pose_csv_files \
  --chunks  path/to/chunk_metadata \
  --tracker tracking_method_name \
  --config  configuration_json_file \
  --cache   path/to/cache
```

To just visualize the cached results, run:

```bash
./scripts/runall --cache path/to/cache --config  configuration_json_file
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

This script converts the input to format that can be read by a tracking algorithm (MOT format)

## (Lane boundary) Tracking:

```bash
./scripts/sort
```

This script applies SORT tracking over detections and saves the results in CSV format

## Evaluation:

```bash
./scripts/evaluate
```

This script applies PYMOT evaluation (multi-object tracking metrics) on the tracking output

## Visualization:

```bash
./scripts/visualize
```

This script displays the output of tracking (e.g. SORT)
