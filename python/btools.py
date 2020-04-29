################################################################
#  Module: btools.py
#  Desc  : Provides methods for handling pribbon matrix
#          operations in parallel
################################################################
#from netCDF4 import Dataset
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
    #          gn      (in): 1d array of global data sizes: (Nz, Ny, Nz)
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
            (ib, ie) = BTools.range(1, self.gn_[2], self.nprocs_, i)
            nxmax = max(nxmax, ie-ib+1)

        szbuff = nxmax*gn[0]*gn[1]
        print(self.myrank_, ": __init__: nxmax=",nxmax," szbuff=",szbuff," gn=",gn)
        sys.stdout.flush()
        buffdims = ([self.comm_.Get_size(), szbuff])
        if   mpiftype == MPI.FLOAT:
            self.recvbuff_ = np.ndarray(buffdims, dtype='f')
        elif mpiftype == MPI.DOUBLE:
            self.recvbuff_ = np.ndarray(buffdims, dtype='d')
        else:
            assert 0, "Input type must be float or double"
        
        self.recvbuff_.fill(self.myrank_)

        linsz = szbuff**2
        print("BTools::__init__: linsz=",linsz)
        sys.stdout.flush()
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
        imax = self.gn_[2]
        jmax = self.gn_[1]
        kmax = self.gn_[0]

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

#       ldata.flatten()

	    # Gather all slabs here to perform thresholding:
     
        print(self.myrank_, ": BTools::buildB: shape(ldata)=", ldata.shape, "shape(recvbuff)=", self.recvbuff_.shape, " ldata.dtype=", ldata.dtype)
        sys.stdout.flush()

        self.comm_.barrier()
        self.comm_.Allgather(ldata,self.recvbuff_)
        self.comm_.barrier()

#       self.recvbuff_[self.myrank_,:] = ldata

        print(self.myrank_, ": BTools::buildB: Allgather done")
        print(self.myrank_, ": BTools::buildB: recvbuff=", self.recvbuff_)
        sys.stdout.flush()

        # Multiply local data by all gathered data and threshold:
        for i in range(0, self.nprocs_):
#           print(self.myrank_, ": BTools::buildB: doing partition ", i, "...")
#           eys.stdout.flush()
            n = self.do_thresh(ldata, self.recvbuff_[i,:], i, cthresh, self.Bp_, self.Ip_, self.Jp_) 
      
            print(self.myrank_, ": BTools::buildB: recvbuff[",i,"]=",self.recvbuff_[i,:])
            sys.stdout.flush()

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
        imax = self.gn_[2]
        jmax = self.gn_[1]
        kmax = self.gn_[0]

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
          ig   = int( float(lnb+i)/float(self.gn_[0]*self.gn_[1]) )
          itmp = ig*self.gn_[0]*self.gn_[1]
          jg   = int( float(lnb+i-itmp)/float(self.gn_[0]) )
          kg   = lnb + i - itmp  - jg*self.gn_[0]

	  # Compute global matrix index: 	    
          Ig = kg + jg*self.gn_[0] + ig*self.gn_[0]*self.gn_[1]

        # print("i=",i," lnb=", lnb, " ig=",ig," jg=",jg,"kg=",kg,": Ig=", Ig)
        # sys.stdout.flush()
          for j in range(0, len(rdata)):
            prod = ldata[i] * rdata[j]
            
            if abs(prod) >= thresh:
     	      # Locate in global grid:
              ig   = int( float(rnb+j)/float(self.gn_[0]*self.gn_[1]) )
              itmp = ig*self.gn_[0]*self.gn_[1]
              jg   = int( float(rnb+j-itmp)/float(self.gn_[0]) )
              kg   = rnb + j - itmp  - jg*self.gn_[0]
           
	      # Compute global matrix indices: 	    
              Jg = kg + jg*self.gn_[0] + ig*self.gn_[0]*self.gn_[1]

        #     print("j=",j," rnb=", rnb, " ig=",ig," jg=",jg,"kg=",kg, ": Ig, Jg=", Ig, Jg)
        #     sys.stdout.flush()
             
              B[n] = prod
              I[n] = int(Ig+0.5)
              J[n] = int(Jg+0.5)
           
              n += 1

#       print(self.myrank_, ": do_thresh: number found=",n, " len(B)=", len(B))
#       sys.stdout.flush()

        return n  # end, do_thresh method
	

    ####################################################
    #  Method: getSlabData
    #  Desc  : Reads specified NetCDF4 file, and returns a slab of data
    #          'owned' by specified MPI rank.
    #  Args  : 
    #          fileName    : string, filename of the netCDF file to open
    #          ensembleName: string, name of the ensemble
    #          itime       : integer, single time (for now,later can be ranges)
    #          mpiTasks    : integer, number of MPI tasks (> 0)
    #          mpiRank     : integer, rank of the calling process [0, mpiTasks-1] 
    #                        (zero based)
    #          means       : integer, 1,2,3 where:
    #                         1: <T(x,y,x)> = Sum ens T(ens,x,y,z)/num ensembles
    #                         2: T(ens,x,y,z) - <T(x,y,z)>
    #                         3: < T(ens,x,y,z) - <T(x,y,z)> >
    #                         4: raw (no subtracted mean)
    #          decimate    : integer, shorten the slab by decimate (0 is no decimate).
    #                        So, if you decimate by 4, you keep every 4th data point
    # ReturnsL N: numpy array, data for a particular mpiRank of 
    #             size (Nx_p, Ny, Nz), where Nx_p is are the number x-planes
    #             corresponding to mpiRank.
    #          gdims: dims of original grid
    ################################################################
    @staticmethod
    def getSlabData(fileName, ensembleName, itime, mpiTasks, mpiRank, means, decimate):
        # N = Btools_getSlabData(fileName, ensembleName, itime, mpiTask, mpiRank, means, decimate)
        # N is a slab of data (x,y,z) 
        #

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
#       nc=Dataset(fileName,'r+',parallel=True)
        nc=Dataset(fileName,'r+')
        N = nc.variables[ensembleName]
        if len(N.shape) != 5:
            sys.exit("Error, ensemble should have five dimensions!")
        gdims = N.shape[2:5]

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
           Nsum = np.zeros([1,iy,ix],dtype=np.float32)
           for i in range(0,iensembles):
              Nsum = Nsum + N[i,itime,1,:,:]
           N = np.true_divide(Nsum,iensembles+1)
           iLstart,iLend = BTools.range(0,ix,mpiTasks, mpiRank)
           gdims = N.shape
           N = N[0,:,iLstart:iLend]
        elif means == 2:  # Subtract the ensemble mean.
           N = nc.variables[ensembleName]
           if len(N.shape) != 5:
              sys.exit("Error, ensemble should have five dimensions!")
           iensembles,ntimes,iz,iy,ix = N.shape
           mean = np.mean(N[0,0,0,:,:])
           gdims = mean.shape
           iLstart,iLend = BTools.range(0,ix,mpiTasks, mpiRank)
           N = N[0,0,0,:,iLstart:iLend] - mean
        elif means == 3:
           N = nc.variables[ensembleName]
           if len(N.shape) != 5:
              sys.exit("Error, ensemble should have five dimensions!")
           iensembles,ntimes,iz,iy,ix = N.shape
           Nsum = np.zeros([1,iy,ix],dtype=np.float32)
           for i in range(0,iensembles):
              Nsum = Nsum + (N[i,itime,1,:,:] - np.mean(N[i,itime,1,:,:]))
           N = np.true_divide(Nsum,iensembles)
           gdims = N.shape
           iLstart,iLend = BTools.range(0,ix,mpiTasks, mpiRank)
           N = N[0,:,iLstart:iLend]
        elif means == 4:
           N = nc.variables[ensembleName]
           if len(N.shape) != 5:
              sys.exit("Error, ensemble should have five dimensions!")
           iensembles,ntimes,iz,iy,ix = N.shape
           iLstart,iLend = BTools.range(0,ix,mpiTasks, mpiRank)
           N = N[0,0,0,:,iLstart:iLend]
           gdims = ([1,iy,ix])
        else:
           sys.exit("Error, bad mean value!")
 
        if decimate != 0:
           print (mpiRank,": getSlabData: N shape=",N.shape)
           sys.stdout.flush()
           N = N[::decimate,::decimate]
           gdims = ([gdims[0],gdims[1], gdims[2]/decimate+1])
 

        nc.close
        print (mpiRank,": getSlabData: N shape_final=",N.shape)
        sys.stdout.flush()
        if len(gdims) == 2:
          gdims = ([1, gdims[0], gdims[1]])
        gdims = ([int(gdims[0]),int(gdims[1]),int(gdims[2]) ])

        return N, gdims
