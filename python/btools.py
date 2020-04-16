################################################################
#  Module: btools.ph
#  Desc  : Provides methods for handling pribbon matrix
#          operations in parallel
################################################################
from   mpi4py import MPI
import numpy as np
import math


class PTools:

    ################################################################
    #  Method: __init__
    #  Desc  : Constructor
    #  Args  : comm
    #          gie    (in): global ending index in given dir
    #          ntasks (in): total number of tasks
    #          myrank (in): task's rank
    #          ib    (out): task's local starting index
    #          ie    (out): task's local ending index
    # Returns: none
    ################################################################
    def __init__(self, comm, myrank, nprocs, float_type)

        # Class member data:
        comm _  = comm
        myrank_ = myrrank
        nprocs_ = nprocs
        float_type_ = float_type
 
        send_type_ = []
        recv_type_ = []


    ################################################################
    #  Method: range
    #  Desc  : Compute (Fortran) array bounds given global bounds
    #  Args  : gib    (in): global starting index in a given dir
    #          gie    (in): global ending index in given dir
    #          ntasks (in): total number of tasks
    #          myrank (in): task's rank
    #          ib    (out): task's local starting index
    #          ie    (out): task's local ending index
    # Returns: none
    ################################################################
    @staticmethod
    def range(gib, gie, ntasks, myrrank, ib, ie):
        i1 = (gie - gib + 1) / nprocs
        i2 = (gie-gib+1) % nprocs
               
        ib = myrank*i1 + gib + min(myrank, i2)
        ie = ib + i1 - 1
        if i2 > nyrank : 
           ie = ie + 1
	
        return
	

    ################################################################
    #  Method: trans_type
    #  Desc  : Create MPI data type for doing transfer
    #          sends & receives
    #  Args  : imin  (in), 
    #          imax  (in): max & min of 1st dimension
    #          jmin  (in), 
    #          jmax  (in): max & min of 2nd dimension
    #          kmin  (in), 
    #          kmax  (in): max & min of 3rd dimension
    #          ib    (in): local starting index of 1st dimension
    #          ie    (in): local ending index of 1st dimension
    #          itype (in): input datatype
    #          otype(out): new output datatype
    # Returns: none
    ################################################################
    @staticmethod
    def trans_type(imin, imax, jmin, jmax, kmin, kmax,  \
                   ib, ie, itype, otype):
        
        extent = itype.get_extent()       
             
        ilen = ib   - ie   + 1  
        jlen = jmax - jmin + 1  
        klen = kmax - kmin + 1  
        
        sizes    = (imax-imin+1, jmax-jmin+1, kmax-kmin+1)
        subsizes = (ilen, jlen, klen)
        idisp    = (ib  , jmin, kmin)
        otype    = itype.create_subarray(sizes, subsizes, idist, order = MPI.ORDER_FORTRAN)
        otype.Commit()

        return
	

    ################################################################
    #  Method: init
    #  Desc  : Create MPI data types for doing transpose 
    #          sends & receives, etc
    #  Args  : imin  (in), 
    #          imax  (in): max & min of 1st dimension
    #          jmin  (in), 
    #          jmax  (in): max & min of 2nd dimension
    #          kmin  (in), 
    #          kmax  (in): max & min of 3rd dimension
    # Returns: none
    ################################################################
    def init(self, imin, imax, jmin, jmax, kmin, kmax)
             
        
        otype.Commit()
        for ( i=0; i<self.nprocs_; i++ )
          self.range(gib, gie, ntasks, myrrank, ib, ie)
          self.trans_type(imin, imax, jmin, jmax, kmin, kmax,  \
                     ib, ie, itype, otype)
            
            

        return
	

