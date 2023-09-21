# Installation

To install AmpTools as a submodule:

```shell
git clone https://github.com/lan13005/pwa-ift --recurse-submodules
```

Environment setup

```shell
conda env create                    # Creates environment specified by environment.yml and pyproject.toml
conda install mpi4py                # For NIFTy to use MPI
conda install root -c conda-forge   # Required for AmpTools
```
