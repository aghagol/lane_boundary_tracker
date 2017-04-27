# preprocess

Data-specific scripts for arranging detections (input) in MOT format. 

Additionally, data-specific steps may be taken to improve the tracking results.

## Input data format

CSV file(s) containing Timestamp-Longitude-Latitude-Altitude (TLLA) points.

Path format: `drive_id/FuseToTLLAIOutput/chunk_number/RANSAC_inlier_subset.csv` where `chunk_number` are integers starting from 0. 

The `--input` option of `runall` script only requires the `drive_id`. Currently, all TLLA points (from all chunks) will be merged in a single array and sorted based on their timestamps regardless of which chunk or `RANSAC _inlier_subset.csv` they come from.

## Output format

 - `tlla.csv`: This is a meta file for later stages of the line connection algorithm
 	+ column 1: `detection's unique identifier`
 	+ column 2: `detection's timestamp`
 	+ column 3: `detection's longitude`
 	+ column 4: `detection's latitude`
 	+ column 5: `detection's altitude`

 - `det.txt`: This is MOT formatted input for the tracking algorithm
 	+ column 1: `detection's frame number`
 	+ column 2: -1 (no label)
 	+ column 3: `detection's pixel row`
 	+ column 4: `detection's pixel column`
 	+ column 5: `detection's bounding box width`
 	+ column 6: `detection's bounding box height`
 	+ column 7: `detection's unique identifier`
 	+ column 8: `detection's timestamp`

