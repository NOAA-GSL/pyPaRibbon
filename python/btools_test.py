import os, sys
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset
import btools

#
# Main Program.
#

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
#(N,gdims) = btools.BTools.getSlabData("Tmerged.nc", "T", 0, mpiTasks, mpiRank, 3, 2)

#
# Instantiate the BTools class before building B:
#
#print ("main: constructing BTools, gdims=",gdims)
#print ("main: constructing BTools, N.shape=",N.shape)
#sys.stdout.flush()

BTools = btools.BTools(comm, MPI.FLOAT, gdims)

# Create some (Fortran-ordered) test data:
ldims  = ([1,2*mpiTasks,2]);
gdims  = ([1, 2, 2*mpiTasks])
Nloc = np.ndarray(ldims,dtype=float,order='C')
Nglo = np.ndarray(gdims,dtype=float,order='C')
print("main: ldims", ldims)
print("main: gdims", gdims)

# Local vector:
istart = myrank*np.prod(ldims)
for k in range(0,ldims[0]):
  for j in range(0,ldims[1]):
    for i in range(0,ldims[2]):
      Nloc[i,j,k] = istart + i + j*ldims[1] + k*ldims[0]*ldims[1]
#
# Global vector:
for k in range(0,gdims[0]):
  for j in range(0,gdims[1]):
    for i in range(0,gdims[2]):
      Nglo[i,j,k] = i + j*gdims[1] + k*gdims[0]*gdims[1]
#
#
# Build the distributed B matrix.
#
B          = []
I          = []
J          = []
threshhold = 0.0
print (mpiRank,": main: calling BTools.buildB...")
sys.stdout.flush()
Nloc = np.asarray(Nloc, order='C')
x = Nloc.flatten()
Nloc = []

BTools.buildB(x, threshhold, B, I, J) 
  
#print (mpiRank, ": len(B)=",len(B))
#print (mpiRank, ": len(I)=",len(I))
#print (mpiRank, ": len(J)=",len(J))

#lcount = len(B)                             # local number of entries
#comm.barrier()
#gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries

#if mpiRank == 0:
#    print(mpiRank, ": main: max number entries  : ", (np.prod(gdims))**2)
#    print(mpiRank, ": main: number entries found: ", gcount)

# TODO: Collect (B,I,J) data from all tasks to task 0, 
#       and plot full matrix, somehow. Compute 'ribbon
#       width'?

# Compute analytic solution:
C = np.tensordot(Nglo.flatten(), Nglo.flatten(), 0)
C[abs(C) < threshold] = 0.


nbad = 0
for i in range(0,len(I)):
    diff = C[I[i],J[i]] - B[i]
    if diff != 0:
        nbad += 1
        print("I=",J[i], " J=",J[i], " diff=", diff)

if nbad > 0:
  print("main: do_threshold FAILED")
else:
  print("main: SUCCESS!")
