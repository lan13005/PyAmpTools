# Installation

To install AmpTools as a submodule:

```shell
git clone https://github.com/lan13005/PyAmpTools --recurse-submodules
```

Environment setup. `conda` can be very slow to resolve dependencies for ROOT. I recommend [Mamba](https://github.com/conda-forge/miniforge#mambaforge)

```shell
conda env create            		# Creates environment specified by environment.yml and pyproject.toml
conda activate PyAmpTools   		# activate the environment
pip install mpi4py                  # MPI, mamba will link it against the wrong executables
mamba install -c conda-forge root   # ROOT!
```

Set some required environment variables

```shell
source set_environment.sh
```

Build AmpTools into a shared library

```shell
cd external/MakeAmpTools/
make # distributes modified Makefiles and makes AmpTools
```
