import numpy as np
import sys
from netCDF4 import Dataset

def BM_ensembleDims(fileName):
   # ix, iy, iz, itime, nens = BM_ensembleDims(fileName)
   # Open netCDF file: fileName and quiry ensemble dimensions.
   #
   # fileName, string, filename of the netCDF file open
   # ix, integer, size of the X coordinate on the ensemble (returned)
   # iy, integer, size of the Y coordinate on the ensemble (returned)
   # iz, integer, size of the Z coordinate on the ensemble (returned)
   # itime, integer, number of time steps recorded (returned)
   # nens, integer, number of ensembles in the fileName (returned)

   if (type(fileName) is not str):
       sys.exit("Error, bad fileName type in BM_ensembleDims!")

   nc=Dataset(fileName,'a')
 
   ix=len(nc.dimensions['west_east'])
   iy=len(nc.dimensions['south_north'])
   iz=len(nc.dimensions['bottom_top'])
   itime=len(nc.dimensions['Time'])
   nens=len(nc.dimensions['ens'])
   nc.close

   return ix,iy,iz,itime,nens

def BM_range(mpiTasks, mpiRank, ix):
   # iLstart,iLend = BM_range(mpiTasks, mpiRank, ix)
   #
   # mpiTasks, integer, number of MPI tasks (> 0)
   # mpiRank, integer, rank of the calling process [0, mpiTasks-1] (zero based)
   # ix, integer, size of the X coordinate of the ensemble
   # iLstart, integer, local start of the slab in the X coordinates (returned)
   # iLend, integer, local end of the slab in the X coordinates (returned)

   if mpiRank <0 or mpiRank>(mpiTasks-1):
       sys.exit("Error, bad mpiRank in BM_range!")

   if (type(mpiRank) is not int):
       print (type(mpiRank))
       sys.exit("Error, bad mpiRank type in BM_range!")

   if type(ix) is not int:
       sys.exit("Error, bad ix type in BM_range!")

   if ix <1:
       sys.exit("Error, bad ix in BM_range!")

   nSlabs = int(ix/mpiTasks)
   iLstart = mpiRank*nSlabs
   iLend   = iLstart + nSlabs - 1
   if mpiRank == (mpiTasks-1):  # Last task takes the overflow.
       iLend = ix-1
   return iLstart,iLend


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
   #    1: <T(x,y,x)> = Sum ens T(ens,x,y,z)/num ensembles
   #    2: T(ens,x,y,z) - <T(x,y,z)>
   #    3: < T(ens,x,y,z) - <T(x,y,z)> >
   #    4: raw (no subtracted mean)
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
   if means == 1: # T(x,y,x) >= Sum ens T(ens,x,y,z)/num ensembles
      N = nc.variables[ensembleName]
      if len(N.shape) != 5:
         sys.exit("Error, ensemble should have five dimensions!")
      iensembles,ntimes,iz,iy,ix = N.shape
      Nsum = np.zeros([1,iy,ix],dtype=float)
      print ("Ensembles = ",iensembles)
      for i in range(0,iensembles):
         Nsum = Nsum + N[i,itime,1,:,:]
      N = np.true_divide(Nsum,iensembles+1)
      iLstart,iLend = BM_range(mpiTasks, mpiRank, ix)
      print ("N shape =", N.shape)
      print ("N.size=",N.size)
      print ("N shape =", N.shape)
      N = N[0,:,iLstart:iLend]
      print ("N shape =", N.shape)
   elif means == 2:  # Subtract the ensemble mean.
      N = nc.variables[ensembleName]
      mean = np.mean(N)
      ix = N.shape[4] # The right most index dimension of N.
      iLstart,iLend = BM_range(mpiTasks, mpiRank, ix)
      N = N[0,0,0,:,iLstart:iLend] - mean
   elif means == 3:
      N = nc.variables[ensembleName]
      ix = N.shape[4] # The right most index dimension of N.
      iLstart,iLend = BM_range(mpiTasks, mpiRank, ix)
      N = N[0,0,0,:,iLstart:iLend]
   else:
      sys.exit("Error, bad mean value!")

   nc.close
   return N

# variables:
# float T(ens, Time, bottom_top, south_north, west_east) ;
# T:FieldType = 104 ;
# T:MemoryOrder = "XYZ" ;
# T:description = "perturbation potential temperature (theta-t0)" ;
# T:units = "K" ;
# T:stagger = "" ;
# T:coordinates = "XLONG XLAT XTIME" ;

#for itasks in range (50,51):
#   for irank in range (3,4):
#      print("irank=",irank,"itasks=",itasks)
#      N = Btools_getSlabData("Tmerged.nc", "T", 0, itasks, irank, 1)
#      print ("N shape =", N.shape)

