
echo "Sourcing additional enviornment variables"

### IF ON IFARM ###
module add cuda
export CUDA_INSTALL_PATH=/apps/cuda/11.4.2/
export GPU_ARCH=sm_75
####################

export REPO_HOME=$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" ) # location of this script

export AMPTOOLS_HOME=$REPO_HOME/external/AmpTools
export AMPTOOLS=$AMPTOOLS_HOME/AmpTools
export AMPPLOTTER=$AMPTOOLS_HOME/AmpPlotter

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH  # for libhwloc.so.5 needed to locate hardware for openmpi
export LD_LIBRARY_PATH=$AMPTOOLS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$AMPTOOLS_HOME/AmpPlotter/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$REPO_HOME/external/AMPTOOLS_AMPS:$REPO_HOME/external/AMPTOOLS_DATAIO:$LD_LIBRARY_PATH

python $REPO_HOME/utils/link_modules.py # symlink files into conda environment
