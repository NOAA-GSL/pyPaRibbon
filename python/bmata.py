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


filepref = "ljunk"

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
(N,nens,gdims) = btools.BTools.getSlabData("Tmerged10.nc", "T", 0, mpiTasks, mpiRank, 2, 1)
if mpiRank == 0:
  print (mpiRank, ": main: constructing BTools, nens   =",nens)
  print (mpiRank, ": main: constructing BTools, gdims  =",gdims)
  print (mpiRank, ": main: constructing BTools, N.shape=",N.shape)
  sys.stdout.flush()

#
# Instantiate the BTools class before building B:
#
#gdims = (1,249,399)
BTools = btools.BTools(comm, MPI.FLOAT, nens, gdims, False)


#
# Build the distributed B matrix.
#
B          = []
I          = []
J          = []
threshold  = 0.8
#print (mpiRank,": main: calling BTools.buildB...")
#sys.stdout.flush()
N = np.asarray(N, order='C')
x=N.flatten()

t0 = time.time()
lcount=BTools.buildB(x, threshold, B, I, J, filename=filepref) 
t1 = time.time()
  
#print (mpiRank, ": len(B)=",len(B))
#print (mpiRank, ": len(I)=",len(I))
#print (mpiRank, ": len(J)=",len(J))
#lcount = len(B)                             # local number of entries

comm.barrier()
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries

ldt = t1 - t0;
gdt = comm.allreduce(ldt, op=MPI.MAX) # global number of entries

#writeResults(B,I,J,"ljunk",mpiRank)
comm.barrier()
if mpiRank == 0:
  print(mpiRank, ": main: max number entries  : ", (np.prod(gdims))**2)
  print(mpiRank, ": main: number entries found: ", gcount)
  print(mpiRank, ": main: data written to file: ", filepref)
  print(mpiRank, ": main: execution time      : ", gdt)
