import os, sys
import glob
import btools
from mpi4py import MPI
from numpy import *
import numpy as np
from numpy import sum
#import datetime
#from datetime import datetime, timedelta
#from scipy import ndimage
#import scipy.stats as stats
#from scipy.optimize import curve_fit
#from scipy.optimize import fmin_cobyla
import math
from netCDF4 import Dataset
import csv
import pandas as pd
from scipy import random
#import pygrib
#from scipy.linalg import eigh as largest_eigh
#from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from time import clock

#from mpl_toolkits.basemap import Basemap
import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import colors

ncfile = "testmerge.nc"

nc=Dataset(ncfile,'a')
pv_b=nc.variables['T']

# Get world size and rank:
comm   = MPI.COMM_WORLD
myrank = comm.Get_rank()

ntime=len(nc.dimensions['Time'])
nx=len(nc.dimensions['west_east'])
ny=len(nc.dimensions['south_north'])
nens=len(nc.dimensions['ens'])
nz=len(nc.dimensions['bottom_top'])
nc.close
#print pv_b.shape
pv_b_n=np.reshape(pv_b,(ntime,nens,nz,nx*ny))
#print pv_b_n.shape

# Get rank's array bounds:



nz=1
pts = zeros(nx*ny)
Bmat=np.zeros([nz,nx*ny,nx*ny],dtype=np.float32)
Bmat_ens=np.zeros([nens,nz,nx*ny,nx*ny],dtype=np.float32)
Bmat_emean=np.zeros([ntime,nz,nx*ny,nx*ny],dtype=np.float32)

##################Calculating B-matrix###############################

for t in range(ntime):
   pv_b_emean=np.mean(pv_b_n[t,:,:,:],0)    #ensemble mean
   #print 'pv_b_mean.shape',pv_b_emean.shape
   #print 'mean',pv_b_emean
   for ens in range(nens):
      pv_b_prime=pv_b_n[t,ens,:,:]-pv_b_emean
      #print 'pv_b_prime.shape',pv_b_prime.shape
      print 'pv_b_prime',pv_b_prime
      pv_b_primeT=np.transpose(pv_b_prime)
      #print 'pvbTshape=',pv_b_primeT.shape
      for lev in range(nz):
         B=np.dot(pv_b_primeT,pv_b_prime)
         #print 'B',B
         #plt.matshow(pv_b_prime);
         #plt.colorbar()
         #plt.matshow(pv_b_primeT);
         #plt.colorbar()
         #plt.matshow(B);
         #plt.colorbar()
         #plt.show()

         #print 'Bshape=',B.shape
         Bmat[lev,:,:]=np.expand_dims(B,0)
         #print 'Bmatshape=',Bmat.shape
         #sys.exit()
      Bmat_ens[ens,:,:,:]=np.expand_dims(Bmat,0)
      #print 'Bmatensshape=',Bmat_ens.shape
   Bmat_emean[t,:,:,:]=np.mean(Bmat_ens,0)
   #print 'Bmatemeanshape=',Bmat_emean.shape
Bclimo=np.mean(Bmat_emean,0)

Bclimo_final=np.squeeze(Bclimo,axis=0)
#print 'Bclimo_final=',Bclimo_final.shape

##############Finished with B-matrix calculation#######################


################converting B-matrix values to cor. coeff.###############
Bclimo_final_final=corrcoef(Bclimo_final)
print 'final_final',Bclimo_final_final
#plt.matshow(Bclimo);
#plt.colorbar()
#plt.show()


################setting to 0 cor. coeff. less than 0.8###############
Bclimo_final[abs(Bclimo_final)<0.8]=0.

#Bclimo_final=ndimage.filters.gaussian_filter(Bclimo_final,1,mode='constant')
#Bclimo_norm[abs(Bclimo_norm)<0.2]=0

#plt.matshow(Bclimo_final);
#plt.colorbar()
#plt.show()


##################determining the width of the ribbon matrix###############

count = []
for y in range (ny*nx):
   countx = 0.
   for x in range (nx*ny):
      if (Bclimo_final[x,y] != 0.):
          countx=countx+1
   count.append(countx)
print 'count',count      
pts=max(count)
print 'pts',pts 

#######################finished with width caluclation##########################

#####################producing netcdf file#######################
##initialize the file`
#nc = Dataset('Bclimo.nc','w', format='NETCDF4_CLASSIC')
#xdim = nc.createDimension('x',99351)
#ydim = nc.createDimension('y',99351)
##zdim = nc.createDimension('z',51)

#x = nc.createVariable('x',np.float32,('x'))
#y = nc.createVariable('y',np.float32,('y'))
##Bc = nc.createVariable('Bclimo',np.float32,('z','y','x'))
#Bc = nc.createVariable('Bclimo',np.float32,('y','x'))


#nc['y'][:]=np.arange(nx*ny)+1
#nc['x'][:]=np.arange(nx*ny)+1
##nc['Bclimo'][:]=Bclimo_norm
#nc['Bclimo'][:]=Bclimo_final
#nc.close
