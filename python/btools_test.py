import os, sys
import numpy as np
from mpi4py import MPI
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
#
#print ("main: constructing BTools, gdims=",gdims)
#print ("main: constructing BTools, N.shape=",N.shape)
#sys.stdout.flush()

#
# Set default local data grid size:
NLx = 1
NLy = 2
NLz = 1

# Create some (Fortran-ordered) test data in order:
# [Nz, Ny, Nz]:
ldims  = ([NLz,NLy, NLx]);         # local dims
gdims  = ([1, NLy, NLx*mpiTasks])  # global dims
Nloc = np.ndarray(ldims,dtype='f',order='C') # local data
Nglo = np.ndarray(gdims,dtype='f',order='C') # global data

print("main: ldims", ldims)
print("main: gdims", gdims)
sys.stdout.flush()


#
# Instantiate the BTools class before building B:
#
BTools = btools.BTools(comm, MPI.FLOAT, gdims)

# Global data:
for k in range(0,gdims[0]):
  for j in range(0,gdims[1]):
    for i in range(0,gdims[2]):
      Nglo[k,j,i] = float(k + j*gdims[0] + i*gdims[0]*gdims[1])

# Local data (taken from global data):
(ib, ie) = btools.BTools.range(1, gdims[2], mpiTasks, mpiRank)
print (mpiRank,": main: ib=", ib, " ie=", ie)
sys.stdout.flush()
for k in range(0,ldims[0]):
  for j in range(0,ldims[1]):
    for i in range(0,ie-ib+1):
      Nloc[k,j,i] = Nglo[k,j,i+ib-1]
#

print (mpiRank,": main: Nloc=", Nloc)
print (mpiRank,": main: Nglo=", Nglo)
sys.stdout.flush()

#
#
# Build the distributed B matrix.
#
B          = []
I          = []
J          = []
threshold = -1.0
print (mpiRank,": main: calling BTools.buildB...")
sys.stdout.flush()

x = Nloc.flatten()
Nloc = []

BTools.buildB(x, threshold, B, I, J) 
print (mpiRank, ": I=",I)
print (mpiRank, ": J=",J)
  
#print (mpiRank, ": len(B)=",len(B))
#print (mpiRank, ": len(I)=",len(I))
#print (mpiRank, ": len(J)=",len(J))

lcount = len(B)                             # local number of entries
comm.barrier()
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries

if mpiRank == 0:
    print(mpiRank, ": main: max number entries  : ", (np.prod(gdims))**2)
    print(mpiRank, ": main: number entries found: ", gcount)

# TODO: Collect (B,I,J) data from all tasks to task 0, 
#       and plot full matrix, somehow. Compute 'ribbon
#       width'?

# Compute analytic solution:
C = np.tensordot(Nglo.flatten(), Nglo.flatten(), 0)
C[abs(C) < threshold] = 0.

print(mpiRank, ": main: C_mat= ", C)
print(mpiRank, ": main: C= ", C.flatten())
print(mpiRank, ": main: B= ", B)



#
# Collect B-matrix entries that are
# wrong:
nbad = 0
for i in range(0,len(B)):
    diff = C[I[i],J[i]] - B[i]
    if abs(diff) > 0:
        nbad += 1
        print(mpiRank, ": I=",I[i], " J=",J[i], " C=", C[I[i],J[i]], " B=", B[i],  " diff=", diff)

if nbad > 0:
  print("main: buildB FAILED: nbad=",nbad)
else:
  print("main: SUCCESS!")
