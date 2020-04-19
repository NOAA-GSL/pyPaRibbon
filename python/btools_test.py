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
ldims  = ([2,2,1]);
gdims  = np.multiply(ldims, nprocs) 
gdims[2] = 1
ldata1 = np.ndarray(ldims,dtype=float,order='F') 
print("main: ldims", ldims)
print("main: gdims", gdims)

seed(10000);
for k in range(0,ldims[2]): 
  for j in range(0,ldims[1]): 
    for i in range(0,ldims[0]): 
      istart = myrank*np.prod(ldims)
      ldata1[i,j,k] = istart + i + j*ldims[1] + k*ldims[0]*ldims[1] + 10*myrank
#     ldata1[i,j,k] = 1.1*random()
#     print("ldata1=",ldata1[i,j,k])


ldata1 = ldata1.flatten()

print("main: ldata1", ldata1)
BTOOLS = btools.BTools(comm, MPI.FLOAT, gdims)

B = []
J = array.array('i')
I = array.array('i')
threshold = 0.0
BTOOLS.buildB(ldata1, threshold, B, I, J)


print("I=")
for j in I: 
    print(j, end=' ')
print('\n')
#
print("J=")
for j in J: 
    print(j, end=' ')
print('\n')
#
print("B=")
for c in B: 
    print(c, end=' ')
print('\n')
#
print("main: max number entries  : ", np.prod(gdims))
print("main: number entries found: ", len(I))

