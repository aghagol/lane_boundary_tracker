## Input format

 - `det.txt`: This is (kind of) MOT formatted input
  + column 1: `detection frame number`
  + column 2: -1 (no label)
  + column 3: `detection pixel row (sub-pixel)`
  + column 4: `detection pixel column (sub-pixel)`
  + column 5: `detection bbox width (may be replaced with row motion)`
  + column 6: `detection bbox height (may be replaced with column motion)`
  + column 7: `detection unique identifier`
  + column 8: `detection timestamp`


## Output format 

 - `seq_name.txt`:
  + column 1: `target frame number`
  + column 2: `target unique identifier`
  + column 3: `target pixel row (sub-pixel)`
  + column 4: `target pixel column (sub-pixel)`
  + column 5: `matched detection's unique identifier` (zero if not matched)
  + column 6: `confidence` (between 0 and 1)
