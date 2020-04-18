################################################################
#  Module: btools.py
#  Desc  : Provides methods for handling pribbon matrix
#          operations in parallel
################################################################
from   mpi4py import MPI
import numpy as np
import math
import sys


class BTools:

    ################################################################
    #  Method: __init__
    #  Desc  : Constructor
    #  Args  : comm   (in): communicator
    #          ftype  (in): MPI float type of data
    #          gn     (in): 1d array of global data sizes
    # Returns: none
    ################################################################
    def __init__(self, comm, ftype, gn):

        # Class member data:
        self.comm_   = comm
        self.myrank_ = comm.Get_rank()
        self.nprocs_ = comm.Get_size()
        self.float_type_ = ftype
 
        self.send_type_ = ftype
        self.recv_type_ = []
        assert len(gn)==3, "Invalid dimension spec"
        self.gn_ = gn
       

    ################################################################
    #  Method: range
    #  Desc  : Compute (Fortran) array bounds given global bounds
    #  Args  : gib    (in): global starting index in a given dir
    #          gie    (in): global ending index in given dir
    #          nprocs (in): total number of tasks
    #          myrank (in): task's rank
    # Returns: (ib ie): task's local starting, ending indices
    ################################################################
    @staticmethod
    def range(gib, gie, nprocs, myrank):

        i1 = (gie - gib + 1) / nprocs
        i2 = (gie - gib + 1) % nprocs
               
        ib = myrank*i1 + gib + min(myrank, i2)
        ie = ib + i1 - 1
        if i2 > myrank: 
           ie = ie + 1
	
        return ib, ie
	

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
    #  Desc  : Create MPI data types, misc data  
    #  Args  : 
    # Returns: none
    ################################################################
    def init(self):
	
        imin = 1
        jmin = 1
        kmin = 1
        imax = self.gn_[0]
        jmax = self.gn_[1]
        kmax = self.gn_[2]

   	#  Create send & receive MPI types:
        (ib, ie) = self.range(imin, imax, self.nprocs_, self.myrank_)
        self.trans_type(imin, imax, jmin, jmax, kmin, kmax,  \
                        ib, ie, itype, send_type_)
        for i in range(0, self.nprocs_):
          (ib, ie) = self.range(imin, imax, self.nprocs_, i)
          self.trans_type(imin, imax, jmin, jmax, kmin, kmax,  \
                     ib, ie, itype, rtype)
        recv_type_[i] = rtype
            

        return
	

    ################################################################
    #  Method: buildB
    #  Desc  : Create 'B-matrix' from distributed data
    #  Args  : ldata  : this task's (local)_ data
    #          cthresh: cov threshold
    #          I, J  : arrays of indieces into global B mat
    #			    where cov > thresh
    # Returns: none
    ################################################################
    def buildB(self, ldata, cthresh, I, J):
	
	# Each task sends to all available larger 
        # task ids, and receives from all available
        # lower task ids.


        # Do thresholding in local covariance members:
        self.thresh(local_data, local_data, cthresh, I, J) 
        
	# Do thresholding during exchange loop;
        # Use blocking sends, receives:
        for i in range(self.myrank_, self.nprocs_):

          if ( self.myrank_ < self.nprocs_-1 ):
            comm_.send(local_data,dest=i+1)

          if ( self.myrank_ > 0 ):
            rdat = comm_.recv(source=i-1)

            self.do_thresh(local_data, rdat, i, cthresh, Ip, Jp) 

          # Append new global indices to return arrays:
          np.append(I, Ip, 0)
          np.append(J, Jp, 0)

        return
	

    ################################################################
    #  Method: do_thresh
    #  Desc  : With local data, and off-task data, compute
    #          global indices where covariance exceeds
    #          specified threshold.
    #  Args  : ldata : this task's (local) data block, assumed 'flattened'
    #          rdata : off-task (remote) data clock, assumed 'flattened'
    #          irecv : task id that rdata is received from
    #          thresh: threshold covariance
    #          I, J  : arrays of indieces into global B mat
    #			    where cov > thresh
    # Returns: 
    ################################################################
    def do_thresh(self, ldata, rdata, irecv, thresh, I, J):
	
        assert len(ldata.shape)==1
        assert len(rdata.shape)==1

        imin = 1
        jmin = 1
        kmin = 1
        imax = self.gn_[0]
        jmax = self.gn_[1]
        kmax = self.gn_[2]

        # Find global starting index of local block:
        (ib, ie) = self.range(imin, imax, self.nprocs_, self.myrank_)
        ib -= 1
        lnb = ib*(jmax-jmin+1)*(kmax-kmin+1)

        # Find global starting index of recv'd block:
        (ib, ie) = self.range(imin, imax, self.nprocs_, irecv)
        ib -= 1
        rnb = ib*(jmax-jmin+1)*(kmax-kmin+1)

	# Order s.t. we multiply
	#    Transpose(ldata) X rdata:
        # wherer Trarnspose(ldata) is a column vector, 
        # and rdata, a row vector in matrix-speak
        n = 0
        for i in range(0, len(ldata)):
     	  # Locate in global grid:
          ig = (lnb+i)/(self.gn_[1]*self.gn_[2])
          jg = (lnb+i-ig)/self.gn_[1]
          kg =  lnb+i - jg*self.gn_[1]
    
	  # Compute global matrix index: 	    
          Ig = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
          for j in range(0, len(rdata)):
            prod = ldata[i] * rdata[j]
            
            if prod >= thresh:
     	      # Locate in global grid:
              ig = (rnb+j)/(self.gn_[1]*self.gn_[2])
              jg = (rnb+j-ig)/self.gn_[1]
              kg =  rnb+j - jg*self.gn_[1]
           
	          # Compute global matrix indices: 	    
              Jg = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
             
              I.append(Ig)
              J.append(Jg)
           
              n += 1

        print("do_thresh: number found=",n)

        return
	

