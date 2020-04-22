################################################################
#  Module: btools.py
#  Desc  : Provides methods for handling pribbon matrix
#          operations in parallel
################################################################
from   mpi4py import MPI
import numpy as np
import array
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

        
        # Build a new communicator for the
	    # GatherV:
        if ( self.myrank_ > 0 ):
           # We include in new comm all ranks from
	   # myrank to nprocs-1 (so, we exclude
	   # from (0, myrank-1):
           group = comm.Get_group()
           pexcl = np.arange(0,self.myrank_,dtype='i')
       	   newgroup = group.Excl(pexcl)
           self.newcomm_ = self.comm_.Create(newgroup)
           group.Free()
           newgroup.Free()
        else:
           self.newcomm_ = comm
 
        self.send_type_ = ftype
        self.recv_type_ = []
        assert len(gn)==3, "Invalid dimension spec"
        self.gn_ = gn
        self.binit_ = 0

        # Create recv buffs for this task:
        nmax = 0
        for i in range(0,self.nprocs_):
            (ib, ie) = BTools.range(1, self.gn_[1], self.nprocs_, i)
        nmax = max(nmax, ie-ib+1)

        if   ftype == MPI.FLOAT:
            nptype = np.float
        elif ftype == MPI.DOUBLE:
            nptype = np.double
        else:
            assert 0, "Input type must be float or double"
        
        szbuff = gn[1]*gn[2]*nmax 
        dims   = ([self.newcomm_.Get_size(), szbuff])
        self.recvbuff_ = np.ndarray(dims, dtype=nptype)

	# end, constructor


    ################################################################
    #  Method: __del__
    #  Desc  : Destructor
    #  Args  : self
    # Returns: none
    ################################################################
#   def __del__(self):       

        #if self.newcomm_: self.newcomm_.Free()
        #self.newcomm_.Free()
	
	# end, destructor



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

        i1 = 0
        i2 = 0
        ib = 0
        ie = 0

        i1 = int((gie - gib + 1) / nprocs)
        i2 = (gie - gib + 1) % nprocs
               
        ib = myrank*i1 + gib + min(myrank, i2)
        ie = ib + i1 - 1
        if i2 > myrank: 
           ie = ie + 1
	
        return ib, ie  # end, range method
	

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

        return # end, trans_type method
	

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
        for i in range(0, self.nprocs_-1):
          (ib, ie) = self.range(imin, imax, self.nprocs_, i)
          self.trans_type(imin, imax, jmin, jmax, kmin, kmax,  \
                     ib, ie, itype, rtype)
        recv_type_[i] = rtype
        self.binit_ = 1       

        return  # end, init method
	

    ################################################################
    #  Method: buildB
    #  Desc  : Create 'B-matrix' from distributed data
    #  Args  : ldata  : this task's (local)_ data
    #          cthresh: cov threshold
    #          B     : array of correlations that exceed threshold
    #          I, J  : arrays of indices into global B mat
    #		       where cov > thresh. Each is of the same length
    # Returns: none
    ################################################################
    def buildB(self, ldata, cthresh, B, I, J):
	
	# Each task sends to all available larger 
        # task ids, and receives from all available
        # lower task ids.


        # Do thresholding in local covariance members:
        self.do_thresh(ldata, ldata, self.myrank_, cthresh, B, I, J) 
        
        # Declare tmp data:
        Bp = []
        Ip = array.array('i')
        Jp = array.array('i')

	# Gather all slabs here to perform thresholding:
     
        print(self.myrank_, ": buildB: buff shape=", np.shape(self.recvbuff_))
        print(self.myrank_, ": buildB: calling Gatherv...")
        print(self.myrank_, ": buildB: ldata=",ldata)
        sys.stdout.flush()
#       self.comm_.Gatherv(ldata, self.recvbuff_, root=self.myrank_)
        recvbuff = self.comm_.allgather(ldata)
        recvbuff = self.recvbuff_
        print(self.myrank_, ": buildB: Gatherv done. recvbuff.shape=", np.shape(recvbuff))
        sys.stdout.flush()


        for i in range(0, self.newcomm_.Get_size()):
            rdata   = self.recvbuff_[i,:]
            srcrank = self.newcomm_.Get_rank()
            self.do_thresh(ldata, rdata, srcrank, cthresh, Bp, Ip, Jp) 

#           print("Bp[",i,"]=")
#           for b in Bp:
#             print(b,Bp,end=' ')
#           print('\n')

            # Append new global indices to return arrays:
            np.append(B, Bp, 0)
            np.append(I, Ip, 0)
            np.append(J, Jp, 0)

        return   # end, buildB method
	

    ####################################################
    #  Method: do_thresh
    #  Desc  : With local data, and off-task data, compute
    #          global indices where covariance exceeds
    #          specified threshold.
    #  Args  : ldata : this task's (local) data block, assumed 'flattened'
    #          rdata : off-task (remote) data clock, assumed 'flattened'
    #          irecv : task id that rdata is received from
    #          thresh: threshold covariance
    #          B     : array of correlations that exceed threshold
    #          I, J  : arrays of indices into global B mat
    #		       where cov > thresh. Each is of the same length
    #                  as B
    # Returns: 
    ################################################################
    def do_thresh(self, ldata, rdata, irecv, thresh, B, I, J):
	
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
          ig   = int( (lnb+i)/(self.gn_[1]*self.gn_[2]) )
          itmp = ig*self.gn_[1]*self.gn_[2]
          jg   = int( (lnb+i-itmp)/self.gn_[1] )
          kg   =  lnb+i - itmp  - jg*self.gn_[1]
          print("i=",i," ig=",ig," jg=",jg,"kg=",kg)

	      # Compute global matrix index: 	    
          Ig = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
          print("Ig=",Ig)
          for j in range(0, len(rdata)):
            prod = ldata[i] * rdata[j]
            
            if prod >= thresh:
     	      # Locate in global grid:
              ig   = int( (rnb+j)/(self.gn_[1]*self.gn_[2]) )
              itmp = ig*self.gn_[1]*self.gn_[2]
              jg   = int( (rnb+j-itmp)/self.gn_[1] )
              kg   =  rnb+j - itmp  - jg*self.gn_[1]
           
	          # Compute global matrix indices: 	    
              Jg = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
             
              B.append(prod)
              I.append(int(Ig))
              J.append(int(Jg))
           
              n += 1

        print("do_thresh: number found=",n)

        return  # end, do_thresh method
	

