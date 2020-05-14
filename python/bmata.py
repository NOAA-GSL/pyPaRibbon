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


filename   = "Tmerged10.nc"
varname    = "T"
threshold  = 0.95
decfact    = 2
soutprefix = "Bmatrix"

# Get world size and rank:
comm     = MPI.COMM_WORLD
mpiTasks = MPI.COMM_WORLD.Get_size()
mpiRank  = MPI.COMM_WORLD.Get_rank()
name     = MPI.Get_processor_name()

print("main: tasks=",mpiTasks, " rank=", mpiRank,"machine name=",name)
sys.stdout.flush()

#
# Get the local data.
#
(N,nens,gdims) = btools.BTools.getSlabData(filename, varname, 0, mpiTasks, mpiRank, 2, decfact)
if mpiRank == 0:
  print (mpiRank, ": main: constructing BTools, nens   =",nens)
  print (mpiRank, ": main: constructing BTools, gdims  =",gdims)
  print (mpiRank, ": main: constructing BTools, N.shape=",N.shape)
  sys.stdout.flush()

#
# Instantiate the BTools class before building B:
#
prdebug = False
BTools = btools.BTools(comm, MPI.FLOAT, nens, gdims, prdebug)

N = np.asarray(N, order='C')
x=N.flatten()
N = None

# Here's where the work is done!
t0 = time.time()
lcount,B,I,J = BTools.buildB(x, threshold) 
x = None
 
# Write out the results.
BTools.writeResults(B, I, J, soutprefix, mpiRank)
 
comm.barrier()
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries

# Compute maximum 'ribbon width':
# First, sort B, I, J on I:
isort = np.argsort(I)
B = B[isort]
I = I[isort]
J = J[isort]

# Find unique row indices:
iu, iunique = np.unique(I, return_index=True)
iu, counts  = np.unique(I, return_counts=True)  
print(mpiRank,": main: len(counts)=",len(counts))
sys.stdout.flush()
iu = None

# For each unique row, find J's and take max(J) - min(J):
ix = np.zeros(np.prod(gdims), dtype=np.int)
Jchk = np.zeros(counts.max(), dtype=np.int)
for j in range(0,len(iunique)):
  for i in range(0,int(counts[j])):
#   print(mpiRank,": main: iunique=",iunique[j], " iunique+i=",iunique[j]+i," len((J)=",len(J))
#   sys.stdout.flush()
#   Jchk[i] = J[iunique[j]+i]
    Jchk[i]  = J[iunique[j]+i]

  nmax    = Jchk.max() - Jchk.min() + 1
  ind     = int(I[iunique[j]])
  ix[ind] = int(nmax)

# Find sum of max width in each row:
print(mpiRank, ": main: Doing global ribbon vector...; ix=", ix)
sys.stdout.flush()
gx  = comm.allreduce(ix, op=MPI.SUM) # Sum of widths over tasks
ribbonWidth = gx.max()
print(mpiRank, ": main: Global ribbon max done.")
sys.stdout.flush()


ldt = time.time() - t0;
gdt = comm.allreduce(ldt, op=MPI.MAX) # global number of entries

comm.barrier()
if mpiRank == 0:
  print(mpiRank, ": main: max number entries  : ", (np.prod(gdims))**2)
  print(mpiRank, ": main: number entries found: ", gcount)
  print(mpiRank, ": main: data written to file: ", soutprefix)
  print(mpiRank, ": main: ribbon width........: ", ribbonWidth)

  print(mpiRank, ": main: execution time      : ", gdt)
