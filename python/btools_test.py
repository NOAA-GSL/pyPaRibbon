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

# Print debug info:
bdebug = False

#
# Get the local data.
#
#(N,gdims) = btools.BTools.getSlabData("Tmerged.nc", "T", 0, mpiTasks, mpiRank, 3, 2)

#
#
# Set default local data grid size:
NLx = 4
NLy = 8
NLz = 1

# Create some (Fortran-ordered) test data in order:
# [Nz, Ny, Nz]:
ldims  = ([NLz,NLy, NLx]);         # local dims
gdims  = ([1, NLy, NLx*mpiTasks])  # global dims
Nloc = np.ndarray(ldims,dtype='f',order='C') # local data
Nglo = np.ndarray(gdims,dtype='f',order='C') # global data

if bdebug:
  print("main: ldims", ldims)
  print("main: gdims", gdims)
  sys.stdout.flush()


#
# Instantiate the BTools class before building B:
#
BTools = btools.BTools(comm, MPI.FLOAT, gdims, bdebug)

# Global data:
for k in range(0,gdims[0]):
  for j in range(0,gdims[1]):
    for i in range(0,gdims[2]):
      Nglo[k,j,i] = float(k + j*gdims[0] + i*gdims[0]*gdims[1])

# Local data (taken from global data):
(ib, ie) = btools.BTools.range(1, gdims[2], mpiTasks, mpiRank)

if bdebug:
  print (mpiRank,": main: ib=", ib, " ie=", ie)
  sys.stdout.flush()

for k in range(0,ldims[0]):
  for j in range(0,ldims[1]):
    for i in range(0,ie-ib+1):
      Nloc[k,j,i] = Nglo[k,j,i+ib-1]
#

if bdebug:
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

if bdebug:
  print (mpiRank,": main: calling BTools.buildB...")
  sys.stdout.flush()

x = Nloc.flatten()
Nloc = []

BTools.buildB(x, threshold, B, I, J) 

if bdebug:
  print (mpiRank, ": I=",I)
  print (mpiRank, ": J=",J)
  

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

#print(mpiRank, ": main: C_mat= ", C)
#print(mpiRank, ": main: C= ", C.flatten())
#print(mpiRank, ": main: B= ", B)

bdims = C.shape
B_mat = np.ndarray(bdims,dtype='f') 
for i in range(0,bdims[0]):
  for j in range(0,bdims[1]):
    B_mat[i,j] = 0.0
#
# Collect B-matrix entries that are
# wrong:
nbad = 0
for i in range(0,len(B)):
    diff = C[I[i],J[i]] - B[i]
    B_mat[I[i],J[i]] = B[i]
    if abs(diff) > 0:
        nbad += 1
#       print(mpiRank, ": I=",I[i], " J=",J[i], " C=", C[I[i],J[i]], " B=", B[i],  " diff=", diff)

if bdebug:
  print(mpiRank, ": main: C_mat=\n", C)
  print(mpiRank, ": main: B_mat=\n", B_mat)
  sys.stdout.flush()

gnbad = comm.allreduce(nbad, op=MPI.SUM) # global max nbad

if mpiRank == 0:
  if gnbad > 0:
    print("\nmain: buildB FAILED: nbad=",gnbad)
  else:
    print("\nmain: SUCCESS!")

comm.barrier()
