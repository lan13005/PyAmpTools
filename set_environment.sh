# launched automatically by conda activate.d
echo "Sourcing additional enviornment variables"

####################
### IF ON JLAB IFARM ###
module add cuda
export CUDA_INSTALL_PATH=/apps/cuda/11.4.2/
export GPU_ARCH=sm_75
####################

# Check if REPO_HOME is a full path OR if REPO_HOME is unset or empty
# VSCode could create additional environment variables... Checking to see if
#   REPO_HOME is a full path is another check to see if it's been set by the user
if [[ ! $REPO_HOME == /* ]] || [ -z "$REPO_HOME" ]; then
    export REPO_HOME=$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" ) # absolute location of this script
    echo "Setting REPO_HOME to $REPO_HOME"
else
    echo "REPO_HOME is already set. Will not attempt override. REPO_HOME=$REPO_HOME"
fi

export AMPTOOLS_HOME=$REPO_HOME/external/AmpTools
export AMPTOOLS=$AMPTOOLS_HOME/AmpTools
export AMPPLOTTER=$AMPTOOLS_HOME/AmpPlotter
export FSROOT=$REPO_HOME/external/FSRoot

## Load Generator library if external/AMPTOOLS_GENERATORS was built
if [ -d "$REPO_HOME/external/AMPTOOLS_GENERATORS/ccdb" ]; then
    # export CCDB_HOME=$REPO_HOME/external/AMPTOOLS_GENERATORS/ccdb
    # source $CCDB_HOME/environment.bash
    export LD_LIBRARY_PATH=$REPO_HOME/external/AMPTOOLS_GENERATORS:$LD_LIBRARY_PATH
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH  # for libhwloc.so.5 needed to locate hardware for openmpi
export LD_LIBRARY_PATH=$AMPTOOLS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$AMPTOOLS_HOME/AmpPlotter/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FSROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$REPO_HOME/external/AMPTOOLS_AMPS_DATAIO:$LD_LIBRARY_PATH

##################### Activate ROOT #################
if [ -f "$REPO_HOME/external/root/thisroot.sh" ]; then
    source $REPO_HOME/external/root/thisroot.sh $REPO_HOME/external/root # setup ROOT
else
    echo ""
    echo "ROOT not found. Please go into root and run build_root.sh to your specifications."
fi

python $REPO_HOME/utility/link_modules.py # symlink C header files so it can be found by ROOT

# Register pa (pyamptools dispatch system) for tab auto-completion of avaiable commands
eval "$(register-python-argcomplete pa)"

# setup auto-load for next time
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d/
