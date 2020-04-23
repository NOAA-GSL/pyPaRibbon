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
    #  Args  : comm    (in): communicator
    #          mpiftype(in): MPI float type of data 
    #          gn      (in): 1d array of global data sizes
    # Returns: none
    ################################################################
    def __init__(self, comm, mpiftype, gn):

        # Class member data:
        self.comm_     = comm
        self.myrank_   = comm.Get_rank()
        self.nprocs_   = comm.Get_size()
        self.mpiftype_ = mpiftype

        self.send_type_ = mpiftype
        self.recv_type_ = []
        assert len(gn)==3, "Invalid dimension spec"
        self.gn_ = gn
        self.binit_ = 0

        # Create recv buffs for this task:
        nxmax = 0
        for i in range(0,self.nprocs_):
            (ib, ie) = BTools.range(1, self.gn_[0], self.nprocs_, i)
            nxmax = max(nxmax, ie-ib+1)

        if   mpiftype == MPI.FLOAT:
            self.npftype_ = np.float
        elif mpiftype == MPI.DOUBLE:
            self.npftype_ = np.double
        else:
            assert 0, "Input type must be float or double"
        
        szbuff = gn[1]*gn[2]*nxmax 
        print(self.myrank_, ": __init__: nxmax=",nxmax," szbuff=",szbuff," gn=",gn)
        sys.stdout.flush()
        self.buffdims_ = ([self.comm_.Get_size(), szbuff])
        self.recvbuff_ = np.ndarray(self.buffdims_, dtype=self.npftype_)
        self.recvbuff_.fill(self.myrank_)

        linsz = szbuff**2
        if   mpiftype == MPI.FLOAT:
            self.Bp_ = array.array('f' , (0.0,)*linsz)
        elif mpiftype == MPI.DOUBLE:
            self.Bp_ = array.array('d' , (0.0,)*linsz)
        else:
            assert 0, "Input type must be float or double"
        self.Ip_ = array.array('i', (0,)*linsz)
        self.Jp_ = array.array('i', (0,)*linsz)

#       print(self.myrank_,": __init__: len(Bp)==",len(self.Bp_))
#       sys.stdout.flush()

	# end, constructor


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
    #          B      : array of correlations that exceed threshold
    #          I, J   : arrays of indices into global B mat
    #		       where cov > thresh. Each is of the same length
    # Returns: none
    ################################################################
    def buildB(self, ldata, cthresh, B, I, J):
	
        print(self.myrank_, ": BTools::buildB: starting...")
        sys.stdout.flush()

	    # Gather all slabs here to perform thresholding:
     
        print(self.myrank_, ": BTools::buildB: len(ldata)=", len(ldata), "len(recvbuff)=", len(self.recvbuff_))
        sys.stdout.flush()

        self.comm_.barrier()
        self.comm_.Allgather(ldata,self.recvbuff_)
        self.comm_.barrier()

        print(self.myrank_, ": BTools::buildB: Allgather done")
        sys.stdout.flush()

        # Multiply local data by all gathered data and threshold:
        for i in range(0, self.nprocs_):
#           print(self.myrank_, ": BTools::buildB: doing partition ", i, "...")
#           eys.stdout.flush()
            n = self.do_thresh(ldata, self.recvbuff_[i,:], i, cthresh, self.Bp_, self.Ip_, self.Jp_) 
#           print(self.myrank_, ": BTools::buildB: partition threshold done.")
#           sys.stdout.flush()

#           print(self.myrank_, ": buildB: Ip[",i,"]=", self.Ip_)
#           print(self.myrank_, ": buildB: Jp[",i,"]=", self.Jp_)
#           print(self.myrank_, ": buildB: Bp[",i,"]=", self.Bp_)
#           sys.stdout.flush()

            # Append new global indices to return arrays:
            B.extend(self.Bp_[0:n])
            I.extend(self.Ip_[0:n])
            J.extend(self.Jp_[0:n])

#           print(self.myrank_, ": buildB: len(l)[",i,"]=", len(ldata),\
#                 " len(r)[",i,"]=",len(self.recvbuff_[i,:]))
#           print(self.myrank_, ": buildB: len(B)[",i,"]=", len(B))
        print(self.myrank_, ": BTools::buildB: partition thresholding done.")
        sys.stdout.flush()


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
    #		       where cov > thresh. We assume that I, J 
    #                  are the same length as B
    #                  as B
    # Returns: number of values found that meet threshold criterion
    ################################################################
    def do_thresh(self, ldata, rdata, irecv, thresh, B, I, J):
	
        assert len(ldata.shape)==1
        assert len(rdata.shape)==1
        assert len(B)==len(I) and len(B)==len(J)

        
        # Just in case, resize operand arrays if necessary:
        if len(ldata)*len(rdata) > len(self.Bp_):
            x = Bp_[0]
            self.Bp_.clear()
            self.Ip_.clear()
            self.Jp_.clear()
            self.Bp_ = [0.0]*(len(ldata)*len(rdata))
            self.Ip_ = [0]  *(len(ldata)*len(rdata)) 
            self.Jp_ = [0]  *(len(ldata)*len(rdata)) 
            

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

#         print("i=",i," ig=",ig," jg=",jg,"kg=",kg)

	      # Compute global matrix index: 	    
          Ig = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
#         print("Ig=",Ig)
          for j in range(0, len(rdata)):
            prod = ldata[i] * rdata[j]
            
            if abs(prod) >= thresh:
     	      # Locate in global grid:
              ig   = int( (rnb+j)/(self.gn_[1]*self.gn_[2]) )
              itmp = ig*self.gn_[1]*self.gn_[2]
              jg   = int( (rnb+j-itmp)/self.gn_[1] )
              kg   =  rnb+j - itmp  - jg*self.gn_[1]
           
	          # Compute global matrix indices: 	    
              Jg = kg + jg*self.gn_[1] + ig*self.gn_[1]*self.gn_[2]
             
              B[n] = prod
              I[n] = int(Ig)
              J[n] = int(Jg)
           
              n += 1

#       print(self.myrank_, ": do_thresh: number found=",n, " len(B)=", len(B))
#       sys.stdout.flush()

        return n  # end, do_thresh method
	

