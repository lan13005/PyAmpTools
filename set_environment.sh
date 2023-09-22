module add cuda
export CUDA_INSTALL_PATH=/apps/cuda/11.4.2/     
export GPU_ARCH=sm_75
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH  # for libhwloc.so.5 needed to locate hardware for openmpi

export IFT_HOME=$(pwd)
