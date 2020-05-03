########################################################################
# Name  : Bmata                                                        #
# Desc  : B-Matrix Analysis program main driver                        #
# Author: L. Stringer                                                  #
#         D. Rosenberg                                                 #
# Date:   April 2020                                                   #
########################################################################
import os, sys
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset
import btools


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
(N,gdims) = btools.BTools.getSlabData("Tmerged.nc", "T", 0, mpiTasks, mpiRank, 3, 0)
print ("main: constructing BTools, gdims=",gdims)
print ("main: constructing BTools, N.shape=",N.shape)
sys.stdout.flush()

#
# Instantiate the BTools class before building B:
#
#gdims = (1,249,399)
BTools = btools.BTools(comm, MPI.FLOAT, gdims, False)


#
# Build the distributed B matrix.
#
B          = []
I          = []
J          = []
threshhold = 0.8
print (mpiRank,": main: calling BTools.buildB...")
sys.stdout.flush()
N = np.asarray(N, order='C')
x=N.flatten()
BTools.buildB(x, threshhold, B, I, J) 
  
print (mpiRank, ": len(B)=",len(B))
print (mpiRank, ": len(I)=",len(I))
print (mpiRank, ": len(J)=",len(J))

lcount = len(B)                             # local number of entries
comm.barrier()
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries
print(mpiRank, ": main: max number entries  : ", (np.prod(gdims))**2)
print(mpiRank, ": main: number entries found: ", gcount)

writeResults(B,I,J,"ljunk",mpiRank)
comm.barrier()
print(mpiRank, ": main: data written to file ", "ljunk.")
