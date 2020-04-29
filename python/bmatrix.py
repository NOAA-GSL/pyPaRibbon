import sys
import re
#
# Interface to the B matrix algorithm.
#
# Here's the command line options:
#    * X Y Z => length of the x/y/z dimensions.
#    * --help => display the command line options
#    * --filename=<filename> => netCDF file containing the ensembles
#    * --decimate=<integer> => increment to decimate the ensemble
#    * --theshhold=<floating point> => threshhold (floating point #)
#

threshhold = .8 # default
decimate = 0    # default
for i in range(len(sys.argv)):
    m = re.match("^--h", sys.argv[i])
    if m:
        print ("python BMatrix nx ny nz --threshhold <fp> --decimate <int>")
        sys.exit(0)
    m = re.match("^--t", sys.argv[i])
    if m:
        threshhold = float(sys.argv[i+1])
    m = re.match("--d", sys.argv[i])
    if m:
        decimate = int(sys.argv[i+1])
    if i==1:
        ix = int(sys.argv[i])
    if i==2:
        iy = int(sys.argv[i])
    if i==3:
        iz = int(sys.argv[i])

print(ix,iy,iz,threshhold,decimate)

