## Input format

 - `det.txt`: This is (kind of) MOT formatted input
 	+ column 1: `detection's frame number`
 	+ column 2: -1 (no label)
 	+ column 3: `detection's pixel row`
 	+ column 4: `detection's pixel column`
 	+ column 5: `detection's bounding box width`
 	+ column 6: `detection's bounding box height`
 	+ column 7: `detection's unique identifier`
 	+ column 8: `detection's timestamp`


## Output format 

 - `seq_name.txt`:
   + column 1: `target's frame number`
   + column 2: `target's unique identifier`
   + column 3: `target's pixel row`
   + column 4: `target's pixel column`
   + column 5: `matched detection's unique identifier` (zero if not matched)
   + column 6: `confidence` (between 0 and 1)
