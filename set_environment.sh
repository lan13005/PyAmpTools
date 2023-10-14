# launched automatically by conda activate.d
echo "Sourcing additional enviornment variables"

conda activate PyAmpTools

####################
### IF ON JLAB IFARM ###
module add cuda
export CUDA_INSTALL_PATH=/apps/cuda/11.4.2/
export GPU_ARCH=sm_75
####################

if [ -z "$REPO_HOME" ]; then
    export REPO_HOME=$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" ) # location of this script
    echo "Setting REPO_HOME to $REPO_HOME"
else
    echo "REPO_HOME is already set. Will not attempt override."
fi

export AMPTOOLS_HOME=$REPO_HOME/external/AmpTools
export AMPTOOLS=$AMPTOOLS_HOME/AmpTools
export AMPPLOTTER=$AMPTOOLS_HOME/AmpPlotter
export FSROOT=$REPO_HOME/external/FSRoot

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH  # for libhwloc.so.5 needed to locate hardware for openmpi
export LD_LIBRARY_PATH=$AMPTOOLS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$AMPTOOLS_HOME/AmpPlotter/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FSROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$REPO_HOME/external/AMPTOOLS_AMPS_DATAIO:$LD_LIBRARY_PATH

##################### Activate ROOT #################
if [ -f "$REPO_HOME/root/thisroot.sh" ]; then
    source $REPO_HOME/root/thisroot.sh $REPO_HOME/root # setup ROOT
else
    echo ""
    echo "ROOT not found. Please go into root and run build_root.sh to your specifications."
fi

python $REPO_HOME/utils/link_modules.py # symlink files into conda environment and root include directory
