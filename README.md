# Installation

Set some required environment variables and setup CUDA 11.4 (JLab ifarm)

```shell
source set_environment.sh
```

To install AmpTools as a submodule:

```shell
git clone https://github.com/lan13005/pwa-ift --recurse-submodules
cd external/AmpTools
make [mpi/gpu/mpigpu] # pick one format. Make sure CUDA_INSTALL_PATH is set
```

Environment setup. `conda` can be very slow to resolve dependencies. I recommend [Mamba](https://github.com/conda-forge/miniforge#mambaforge)

```shell
conda env create                            # Creates environment specified by environment.yml and pyproject.toml
conda activate pwa-ift                      # activate the environment
mamba install mpi4py                        # For NIFTy to use MPI
mamba install -c conda-forge root=6.24.04   # cuda 11.4 requires gcc < 11. Install root v6.24 (matching ifarm) which requires python <3.9
```
