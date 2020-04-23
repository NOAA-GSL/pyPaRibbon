import os, sys
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset

import btools

def Btools_getSlabData(fileName, ensembleName, itime, mpiTasks, mpiRank, means):
   # N = Btools_getSlabData(fileName, ensembleName, itime, mpiTask, mpiRank, means)
   # N is a slab of data (x,y,z) 
   #
   # fileName, string, filename of the netCDF file to open
   # ensembleName, string, name of the ensemble
   # itime, integer, single time (for now,later can be ranges)
   # mpiTasks, integer, number of MPI tasks (> 0)
   # mpiRank, integer, rank of the calling process [0, mpiTasks-1] (zero based)
   # means, integer, 1,2,3 where:
   #    1: T(x,y,x) >= Sum ens T(ens,x,y,z)/num ensembles
   #    2: N - mean of ensembleName
   #    3: raw (no subtracted mean)
   # N, numpy array, data for a particular mpiRank

   #
   # Check the inputs.
   #
   if mpiRank <0 or mpiRank>(mpiTasks-1):
       sys.exit("Error, bad mpiRank in Btools_getSlabData!")

   if (type(mpiRank) is not int):
       sys.exit("Error, bad mpiRank type in Btools_getSlabData!")
   
   if (type(itime) is not int):
       sys.exit("Error, bad itime type in Btools_getSlabData!")

   if (type(fileName) is not str):
       sys.exit("Error, bad fileName type in Btools_getSlabData!")

   # 
   # Open the netCDF file, what is read depends on means.
   #
   nc=Dataset(fileName,'a')

   #
   # Return the selected data.
   #
   #    1: <T(x,y,x)> = Sum ens T(ens,x,y,z)/num ensembles
   #    2: T(ens,x,y,z) - <T(x,y,z)>
   #    3: < T(ens,x,y,z) - <T(x,y,z)> >
   #    4: raw (no subtracted mean)

   if means == 1: # T(x,y,x) >= Sum ens T(ens,x,y,z)/num ensembles
      N = nc.variables[ensembleName]
      if len(N.shape) != 5:
         sys.exit("Error, ensemble should have five dimensions!")
      iensembles,ntimes,iz,iy,ix = N.shape
      Nsum = np.zeros([1,iy,ix],dtype=float)
      for i in range(0,iensembles):
         Nsum = Nsum + N[i,itime,1,:,:]
      N = np.true_divide(Nsum,iensembles+1)
      iLstart,iLend = BTools_range(mpiTasks, mpiRank, ix)
      N = N[0,:,iLstart:iLend]
   elif means == 2:  # Subtract the ensemble mean.
      N = nc.variables[ensembleName]
      if len(N.shape) != 5:
         sys.exit("Error, ensemble should have five dimensions!")
      iensembles,ntimes,iz,iy,ix = N.shape
      mean = np.mean(N[0,itimes,0,:,:])
      iLstart,iLend = BTools_range(mpiTasks, mpiRank, ix)
      N = N[0,0,0,:,iLstart:iLend] - mean
   elif means == 3:
      N = nc.variables[ensembleName]

   elif means == 4:
      N = nc.variables[ensembleName]
      ix = N.shape[4] # The right most index dimension of N.
      iLstart,iLend = BTools_range(mpiTasks, mpiRank, ix)
      N = N[0,0,0,:,iLstart:iLend]
   else:
      sys.exit("Error, bad mean value!")

   nc.close
   print ("N shape =", N.shape)
   return N

def BTools_range(mpiTasks, mpiRank, ix):
   # iLstart,iLend = BTools_range(mpiTasks, mpiRank, ix)
   #
   # mpiTasks, integer, number of MPI tasks (> 0)
   # mpiRank, integer, rank of the calling process [0, mpiTasks-1] (zero based)
   # ix, integer, size of the X coordinate of the ensemble
   # iLstart, integer, local start of the slab in the X coordinates (returned)
   # iLend, integer, local end of the slab in the X coordinates (returned)

   if mpiRank <0 or mpiRank>(mpiTasks-1):
       sys.exit("Error, bad mpiRank in BTools_range!")

   if (type(mpiRank) is not int):
       sys.exit("Error, bad mpiRank type in BTools_range!")

   if type(ix) is not int:
       sys.exit("Error, bad ix type in BTools_range!")

   if ix <1:
       sys.exit("Error, bad ix in BTools_range!")

   nSlabs = int(ix/mpiTasks)
   iLstart = mpiRank*nSlabs
   iLend   = iLstart + nSlabs - 1
   if mpiRank == (mpiTasks-1):  # Last task takes the overflow.
       iLend = ix-1
   print ("range iLstart =",iLstart,"iLend=",iLend)
   sys.stdout.flush()
   return iLstart,iLend

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
N = Btools_getSlabData("Tmerged.nc", "T", 0, mpiTasks, mpiRank, 4)

#
# Substantiate the Btools class.
#
gdims = np.array([399,249,1])
print ("main: constrructing BTools, gdims=",gdims)
sys.stdout.flush()
BTools = btools.BTools(comm, MPI.FLOAT, gdims)

#
# Build the distributed B matrix.
#

B          = []
I          = []
J          = []
threshhold = 0.8
N = N.flatten()
print ("main: calling BTools.buildB...")
sys.stdout.flush()
BTools.buildB(N, threshhold, B, I, J) 

print ("len(B)=",len(B))
print ("len(I)=",len(I))
print ("len(J)=",len(J))

lcount = len(B)                             # local number of entries
gcount = comm.allreduce(lcount, op=MPI.SUM) # global number of entries
print(myrank, ": main: max number entries  : ", (np.prod(gdims))**2)
print(myrank, ": main: number entries found: ", gcount)

# TODO: Collect (B,I,J) data from all tasks to task 0, 
#       and plot full matrix, somehow. Compute 'ribbon
#       width'?

