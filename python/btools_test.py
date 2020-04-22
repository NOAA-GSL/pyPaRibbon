import os, sys
from mpi4py import MPI
from numpy import *
import numpy as np
from numpy import sum
import math
from random import seed
from random import random
import array
import btools


# Get world size and rank:
comm   = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()
print("main: nprocs=",nprocs, " rank=", myrank)

# Create some (Fortran-ordered) test data:
ldims  = ([2,2*nprocs,1]);
gdims  = ([2*nprocs,ldims[1],1])

ldata1 = np.ndarray(ldims,dtype=float,order='F') 
print(myrank, ": main: ldims", ldims)
print(myrank, ": main: gdims", gdims)
sys.stdout.flush()

seed(10000);
for k in range(0,ldims[2]): 
  for j in range(0,ldims[1]): 
    for i in range(0,ldims[0]): 
      istart = myrank*np.prod(ldims)
      ldata1[i,j,k] = istart + i + j*ldims[1] + k*ldims[0]*ldims[1] + 10*myrank
#     ldata1[i,j,k] = 1.1*random()
#     print("ldata1=",ldata1[i,j,k])


ldata1 = ldata1.flatten()

print(myrank, ": main: ldata1", ldata1)
sys.stdout.flush()
BTOOLS = btools.BTools(comm, MPI.FLOAT, gdims)

B = []
J = array.array('i')
I = array.array('i')
threshold = -1.0

print(myrank, ": main: calling buildB...")
sys.stdout.flush()
BTOOLS.buildB(ldata1, threshold, B, I, J)


print(myrank, ": main: I=", I)
print(myrank, ": main: J=", J)
print(myrank, ": main: B=", B)

lnumber = len(B)                               # local number of entries
#gnumber = comm.allreduce(lnumber, op=MPI.SUM) # global number of entries
gnumber = lnumber
print(myrank, ": main: max number entries  : ", (np.prod(gdims))**2)
print(myrank, ": main: number entries found: ", len(B))

# TODO: Collect (B,I,J) data from all tasks to task 0, 
#       and plot full matrix, somehow. Compute 'ribbon
#       width'?

