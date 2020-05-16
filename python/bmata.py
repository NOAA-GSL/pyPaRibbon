########################################################################
# Name  : Bmata                                                        #
# Desc  : B-Matrix Analysis program main driver                        #
# Author: L. Stringer                                                  #
#         D. Rosenberg                                                 #
# Date:   April 2020                                                   #
########################################################################
import os, sys
import numpy as np
from   mpi4py import MPI
from   netCDF4 import Dataset
import time
import btools

# User specifiable data:
filename   = "Tmerged17.nc" # input file
varname    = "T"            # input file variable name
threshold  = 0.95           # correl. coeff thrreshold
decfact    = 8              # 'decimation factor' in x, y directions
soutprefix = "Bmatrix"      # B matrix output prefix

# Get world size and rank:
comm     = MPI.COMM_WORLD
mpiTasks = MPI.COMM_WORLD.Get_size()
mpiRank  = MPI.COMM_WORLD.Get_rank()
name     = MPI.Get_processor_name()

print("main: tasks=",mpiTasks, " rank=", mpiRank,"machine name=",name)
sys.stdout.flush()

# Get the local data:
(N,nens,gdims) = btools.BTools.getSlabData(filename, varname, 0, mpiTasks, mpiRank, 2, decfact)
if mpiRank == 0:
  print (mpiRank, ": main: constructing BTools, nens   =",nens)
  print (mpiRank, ": main: constructing BTools, gdims  =",gdims)
  print (mpiRank, ": main: constructing BTools, N.shape=",N.shape)
  sys.stdout.flush()

# Instantiate the BTools class before building B:
prdebug = False
BTools = btools.BTools(comm, MPI.FLOAT, nens, gdims, prdebug)

N = np.asarray(N, order='C')
x=N.flatten()
N = None

# Here's where the work is done!
t0 = time.time()
lcount,B,I,J = BTools.buildB(x, threshold) 
x = None
 
# Write out the results:
BTools.writeResults(B, I, J, soutprefix, mpiRank)
 
comm.barrier()
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries

# Compute 'ribbon widths':
# First, sort B, I, J on I:
isort = np.argsort(I)
B = B[isort]
I = I[isort]
J = J[isort]

if mpiRank == 0:
  print(mpiRank, ": main: sort(I)=", I)
  print(mpiRank, ": main: sort(J)=", J)

# Next, find unique row indices:
iu, iunique = np.unique(I, return_index=True)
iu, counts  = np.unique(I, return_counts=True)  
print(mpiRank,": main: len(counts)=",len(counts))
sys.stdout.flush()
iu = None

# Then, for each unique row, find J's and find 
# 'local' ribbon width by taking max(J) - min(J):
#ilen  = np.zeros(np.prod(gdims), dtype='i')
Jmax  = np.zeros(np.prod(gdims), dtype='i') #np.int)
Jmin  = np.zeros(np.prod(gdims), dtype='i') #np.int)

IMAX  = np.prod(gdims)+10
Jmax.fill(-1)
Jmin.fill(IMAX)
for i in range(0,len(iunique)):
  jjmax = -1
  jjmin = IMAX
  for j in range(0,int(counts[i])):
#   print(mpiRank,": main: iunique=",iunique[j], " iunique+i=",iunique[j]+i," len((J)=",len(J))
#   sys.stdout.flush()
    Jchk   = J[iunique[i]+j]
    jjmax  = max(jjmax, Jchk)
    jjmin  = min(jjmin, Jchk)

# lwidth    = jjmax - jjmin + 1
  ind       = int(I[iunique[i]])
  Jmax[ind] = jjmax
  Jmin[ind] = jjmin
# ilen[ind] = int(lwidth)


# Find sum of 'local' widths in each row:
#print(mpiRank, ": main: Doing global ribbon vector...; ix=", ix)
#sys.stdout.flush()
#glen  = np.zeros(np.prod(gdims), dtype='i')
#comm.Allreduce(ilen, glen, op=MPI.SUM) # Sum of widths over tasks

print(mpiRank, ": main: Jmax=", Jmax[0:100])
print(mpiRank, ": main: Jmin=", Jmin[0:100])
sys.stdout.flush()


gJmax = np.zeros(np.prod(gdims), dtype='i') #np.int)
comm.Allreduce(Jmax, gJmax, op=MPI.MAX) # Sum of widths over tasks
Jmax = None
gJmin = np.zeros(np.prod(gdims), dtype='i') #np.int)
comm.Allreduce(Jmin, gJmin, op=MPI.MIN) # Sum of widths over tasks
Jmin = None

if gJmax.max() >= np.prod(gdims): 
  print(mpiRank, ": main: gJmax.max=", gJmax.max())
  sys.stdout.flush()
  sys.exit("Invalid index in gJmax")
if gJmin.min() < 0: 
  print(mpiRank, ": main: gJmin.max=", gJmin.max(), " gJmin.min=", gJmin.min())
  sys.stdout.flush()
  sys.exit("Invalid index in gJmin")

# Compute ribbon width for each row of B-matrix:
gJmax -= gJmin
ribbonWidth = gJmax.max()
irowmax = np.argmax(gJmax)
#print(mpiRank, ": main: Global ribbon max done.")
#sys.stdout.flush()

# Write width distribution to a file:
gJmax[gJmax < 0] = 0 
wfilename = soutprefix + "." + "width" + "." + str(threshold) + "." + str(decfact) + ".txt"
np.savetxt(wfilename, gJmax, delimiter="\n")


# Compute total run time:
ldt = time.time() - t0;
gdt = comm.allreduce(ldt, op=MPI.MAX) # global number of entries

comm.barrier()

if mpiRank == 0:
  sfilename = soutprefix + "." + "summary" + "." + str(threshold) + "." + str(decfact) + ".txt"
  f = open(sfilename,'w')
  f.write("main: max number entries ...... : %d\n"% (np.prod(gdims))**2)
  f.write("main: number entries > threshold: %d\n"% gcount)
  f.write("main: data written to file......: %s\n"% soutprefix)
  f.write("main: max ribbon width..........: %d\n"% np.prod(gdims))
  f.write("main: ribbon width..............: %d\n"% ribbonWidth)
  f.write("main: row of ribbon width.......: %d\n"% irowmax)
  f.write("main: execution time............: %f\n"% gdt)
  f.close()
  print(mpiRank, ": main: max number entries ...... : ", (np.prod(gdims))**2)
  print(mpiRank, ": main: number entries > threshold: ", gcount)
  print(mpiRank, ": main: data written to file......: ", soutprefix)
  print(mpiRank, ": main: max ribbon width..........: ", np.prod(gdims))
  print(mpiRank, ": main: ribbon width..............: ", ribbonWidth)
  print(mpiRank, ": main: row of ribbon width.......: ", irowmax)
  print(mpiRank, ": main: execution time............: ", gdt)
