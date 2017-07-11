# lane_boundary_tracker

Bash and Python scripts for preprocessing, processing (tracking), visualization and evaluation for lane boundary tracking algorithm.

## Usage

To execute the complete process, run:

```bash
scripts/runall  --images    path/to/CNN_predictions \
                --drives    list_of_drives_txt_file \
                --poses     path/to/pose_csv_files \
                --chunks    path/to/chunk_metadata \
                --tracker   tracking_method_name \
                --config    configuration_json_file \
                --verbosity verbosity level (0,1,2), default is 1 \
                --visualize visualize (0=NO,1=YES), default is 1 \
                --cache     path/to/cache \
                --fuses     path/to/fuse_files \
                --tagged    path/to/tagged_fuse_files
```

 - A sample configuration file `conf.json` is located at the repository root.
 - The `list_of_drives_txt_file` contains IDs of drives to be processed.
 - If the `--drives` option is not used, a `drive_list.txt` is automatically generated.
 - If the `--drives` input file does not point to an existing file, it is generated from images.
 - `pose` and `chunk` files (drive metadata) are named as `driveID-pose.csv` and `driveID.csv` respectively.
 - `fuses` and `tagged` paths are separated from `cache` path since, unlike cache, they are reusable. 
 

## Installation

I have tested the code with Anaconda 4.3.1 (Python 2.7.13 64-bit). 

Python packages needed for running the code are:

 - `numpy`
 - `scipy`
 - `pandas`
 - `networkx`
 - `pillow`
 - `matplotlib` (for visualization)
 - `scikit-learn`
 - `numba` (included in Anaconda)
 - `json`
 - `jsmin`
 - `filterpy`

Most (if not all) of these packages can be installed simply using `pip install package_name`.
