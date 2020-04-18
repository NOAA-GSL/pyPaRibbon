import os, sys
import btools
from mpi4py import MPI
from numpy import *
import numpy as np
from numpy import sum
import math

ncfile = "testmerge.nc"

nc=Dataset(ncfile,'a')
pv_b=nc.variables['T']

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

ldata1.flatten()
ldata2.flatten()

BTOOLS = pBTools(comm, MPI.FLOAT, gdims)

BTOOLS.do_thresh(ldata1, ldata2, 0, 10.0, I, J)

I
J


