import os, sys
from mpi4py import MPI
from numpy import *
import numpy as np
from numpy import sum
import math
import btools


# Get world size and rank:
comm   = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

# Create some test data:
ldims  = ([2,2,1]);
gdims  = ldims * nprocs 
ldata1 = np.ndarray(ldims,dtype=float,order='F') 

for k in range(0,gdims[2]): 
  for j in range(0,gdims[1]): 
    for i in range(0,gdims[0]): 
      ldata1[i,j,k] = i + j*gdims[1] + k*gdims[0]*gdims[1] + myrank

ldata2 = ldata1 * nprocs

ldata1 = ldata1.flatten()
ldata2 = ldata2.flatten()

print("main: ldata1.shape=",ldata1.shape)

BTOOLS = btools.BTools(comm, MPI.FLOAT, gdims)

J = []
I = []
BTOOLS.do_thresh(ldata1, ldata2, 0, 10.0, I, J)

print("I=")
for j in I: 
    print(j,)

print("J=")
for j in J: 
    print(j,)

