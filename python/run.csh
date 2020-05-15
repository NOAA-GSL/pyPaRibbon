#!/bin/tcsh
#SBATCH -A gsd-hpcs
#SBATCH -q batch
#SBATCH -o gtest

## NOTE: 20 FGE cores, 8 GPUs on FGE
##       22 theia cores per node
##       24 hera cores per node
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00

setenv USE_GPUS 0

cd ${SLURM_SUBMIT_DIR}

module use /scratch2/BMC/gsd-hpcs/Christopher.W.Harrop/bass-develop/modulefiles
module load BASS
module load Anaconda3
module load gcc
module load impi 

setenv I_MPI_PIN         disable

# Number of CPU cores
setenv CPU_CORES_PER_RANK 1

setenv CUDA_DEVICE_MAX_CONNECTIONS 12
setenv CUDA_COPY_SPLIT_THRESHOLD_MB 1

# Using impi, must set the following to
# see the filesystem:
setenv I_MPI_EXTRA_FILESYSTEM on
setenv I_MPI_EXTRA_FILESYSTEM_LIST lustre:panfs

# KMP_AFFINITY controls affinity with impi:
setenv KMP_AFFINITY scatter

setenv NP   $SLURM_NTASKS
setenv NPPN $SLURM_TASKS_PER_NODE
#setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK

if ( $NPPN == 2 ) then
  setenv OMP_NUM_THREADS 10
else if ( $NPPN == 4 ) then
  setenv OMP_NUM_THREADS 4
else if ( $NPPN == 8 ) then
  setenv OMP_NUM_THREADS 2
else
##@ OMP_NUM_THREADS = 20 / $NPPN
setenv OMP_NUM_THREADS 20
endif

# Following used for use on FGA system due to
# use different IB cards than on reg theia nodes:
setenv I_MPI_FABRICS shm:ofa

# Set GPU ids, if necessary:
if ( $USE_GPUS == 1 ) then
  if ( $NP <= 4 ) then
    setenv CUDA_VISIBLE_DEVICES "0,1,2,3"
  else
    setenv CUDA_VISIBLE_DEVICES "0,1,2,3,4,5,6,7"
  endif

  if ( $NPPN > 8 ) then
    echo "Must have 1 MPI task bound to 1 GPU!"
  endif

endif

echo "NP=" $NP
echo "NPPN=" $NPPN
echo "OMP_NUM_THREADS=" $OMP_NUM_THREADS

module list
scontrol show hostname $SLURM_NODELIST#  exit

srun -n $NP  python ./bmata.py

