echo ""
echo "*******************"
echo "Loading environment"
echo "*******************"
echo ""

env_name="pyamptools"
default_env="/w/halld-scshelf2101/lng/WORK/PyAmpTools" # Default env location on Jlab farm. Will be compared to PWD so no trailing slash!


# VSCode could create additional environment variables...
#   Checking to see if PYAMPTOOLS_HOME is a full path to see if it's been set by the user
if [[ ! $PYAMPTOOLS_HOME == /* ]] || [ -z "$PYAMPTOOLS_HOME" ]; then
    export PYAMPTOOLS_HOME=$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" ) # absolute location of this script
    echo "PYAMPTOOLS_HOME was set to $PYAMPTOOLS_HOME"
# else
#     echo "PYAMPTOOLS_HOME is already set. Will not attempt override. PYAMPTOOLS_HOME=$PYAMPTOOLS_HOME"
fi

####################
# Check if the hostname contains "jlab.org" if so we perform default setup
#    and link external libraries against a central repo (its actually my
#    personal working directory but it isn't modified much anymore)

hostname=$(hostname)
if [[ "$hostname" == *"jlab.org"* ]]; then

    echo "Hostname contains 'jlab.org'. Loading default JLab environment..."
    echo ""

    # Was in .bashrc is this good place now?
    source /etc/profile.d/modules.sh
    module use /apps/modulefiles
    module load mpi/openmpi3-x86_64
    module load gcc/9.3.0
    module load cuda

    export CUDA_INSTALL_PATH=/apps/cuda/11.4.2/
    export GPU_ARCH=sm_75
    export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH # needed for mpi4py

    # You could build the external libraries from source but since you are
    # on the JLab system you can also just use the pre-built libraries
    if [[ ! "$PYAMPTOOLS_HOME" == "$default_env" ]]; then
        mv $PYAMPTOOLS_HOME/external $PYAMPTOOLS_HOME/.external # hide current external directory
        ln -s "$default_env/external" $PYAMPTOOLS_HOME # link over the pre-built libraries
    fi
fi
####################

# Perform some checks if you are currently in the base conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "You are currently in the base conda environment (or conda doesn't exist at all)"
    if conda env list | grep -q "$env_name"; then
        echo "Please activate '$env_name' environment and SOURCE AGAIN"
    else
        # of course I can create and activate the env here but the user should manually do it (I think)
        echo "Conda environment '$env_name' does not exist. Please create it, activate it, and SOURCE AGAIN"
    fi
    echo ""
    return
fi

### Set up environment variables for AmpTools ###
export AMPTOOLS_HOME=$PYAMPTOOLS_HOME/external/AmpTools
export AMPTOOLS=$AMPTOOLS_HOME/AmpTools
export AMPPLOTTER=$AMPTOOLS_HOME/AmpPlotter
export FSROOT=$PYAMPTOOLS_HOME/external/FSRoot

## Load Generator library if external/AMPTOOLS_GENERATORS was built
if [ -d "$PYAMPTOOLS_HOME/external/AMPTOOLS_GENERATORS/ccdb" ]; then
    # export CCDB_HOME=$PYAMPTOOLS_HOME/external/AMPTOOLS_GENERATORS/ccdb
    # source $CCDB_HOME/environment.bash
    export LD_LIBRARY_PATH=$PYAMPTOOLS_HOME/external/AMPTOOLS_GENERATORS:$LD_LIBRARY_PATH
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH  # for libhwloc.so.5 needed to locate hardware for openmpi
export LD_LIBRARY_PATH=$AMPTOOLS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$AMPTOOLS_HOME/AmpPlotter/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FSROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYAMPTOOLS_HOME/external/AMPTOOLS_AMPS_DATAIO:$LD_LIBRARY_PATH


##################### Activate ROOT #################
if [ -f "$PYAMPTOOLS_HOME/external/root/thisroot.sh" ]; then
    source $PYAMPTOOLS_HOME/external/root/thisroot.sh $PYAMPTOOLS_HOME/external/root # setup ROOT
else
    echo ""
    echo "ROOT not found. Please go into root and run build_root.sh to your specifications."
fi

python $PYAMPTOOLS_HOME/utility/link_modules.py # symlink C header files so it can be found by ROOT

# Register pa (pyamptools dispatch system) for tab auto-completion of avaiable commands
eval "$(register-python-argcomplete pa)"

# setup auto-load for next time
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/

# readlink -f can still return a location even if set_environment.sh is not in the current directory
#    lets check that it is a valid file first
if [ -e "$(readlink -f set_environment.sh)" ]; then
    ln -sf "$(readlink -f set_environment.sh)" $CONDA_PREFIX/etc/conda/activate.d/
fi
