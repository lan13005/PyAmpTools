# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools). Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. There are no known features of AmpTools that is not currently supported and both GPU and MPI is working. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

Features:

- Access to PyROOT ecosystem
- [Pythonization](https://root.cern/manual/python/#pythonizing-c-user-classes) of c++ objects: simplify interactions with c++ source code
- Dynamically load appopriate libraries / (re)compilation of high level scripts (like fits and plotters) are time consuming and distracting
- Python ecosystem:
  - Improved scripting, string parsing (regex)
  - Markov Chain Monte Carlo: [emcee](https://emcee.readthedocs.io/en/stable/)
  - ...

# Installation

To install AmpTools and FSRoot as a submodule:

```shell
git clone https://github.com/lan13005/PyAmpTools --recurse-submodules
```

Environment setup. `conda` can be very slow to resolve dependencies for ROOT. I recommend [Mamba](https://github.com/conda-forge/miniforge#mambaforge)

```shell
conda env create  # Creates environment specified by environment.yml and pyproject.toml
conda activate PyAmpTools # activate the environment
pip install mpi4py # MPI (if available), mamba will link it against the wrong executables
pre-commit install --install-hooks # (Optional) commit hooks to perform loose formatting
```

[ROOT](https://root.cern/install/) is a required dependency. There is a known conflict between AmpTools' GPU usage and RooFit/TMVA which comes with the conda-forge binaries of ROOT. Currently, ROOT has to be built from source with roofit and tmva off. A build script is included to download ROOT from source with the appropriate cmake flags to achieve this

```shell
cd root
# Modify build_root.sh to match your environment (gcc versions, etc))
source build_root.sh
```

Modify `set_environment.sh` to match you GPU environment (default: setup for JLab ifarm)

```shell
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d # load environment variables on conda activation
```

Build required libraries

```shell
# Options in brackets are optional
cd external
make [mpi/gpu/mpigpu/gpumpi] # distributes modified Makefiles and makes AmpTools + AMPS_DATAIO into a shared libraries
```

Simple Unit tests

```shell
sed -i "s~REPLACE_FOLDER_LOCATION~$REPO_HOME/tests/samples/SIMPLE_EXAMPLE~" $REPO_HOME/tests/samples/SIMPLE_EXAMPLE/fit.cfg # update path
pytest -v
pytest -k [marked-test] # to run a specific marked test defined in pytest.ini
```
