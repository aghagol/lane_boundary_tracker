## output format 

 - `seq_name.txt`: This is MOT formatted input for the tracking algorithm
   + column 1: `target's frame number`
   + column 2: `target's unique identifier`
   + column 3: `target's pixel row`
   + column 4: `target's pixel column`
   + column 5: `matched detection's unique identifier` (zero if not matched)
   + column 6: `confidence` (between 0 and 1)
