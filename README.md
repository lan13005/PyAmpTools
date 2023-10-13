# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools). Under the hood, it uses PyROOT which is based on cppyy. These bindings will
hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem.

Features:

- Access to [PyROOT](https://root.cern/manual/python/) ecosystem
- [Pythonization](https://root.cern/manual/python/#pythonizing-c-user-classes) of c++ objects: simplify interactions with c++ source code
- Dynamically load appopriate libraries / (re)compilation of high level scripts (like fits and plotters) are time consuming and distracting
- Python ecosystem:
  - Improved scripting, string parsing (regex)
  - Markov Chain Monte Carlo: [emcee](https://emcee.readthedocs.io/en/stable/)
  - ...

# Installation

To install AmpTools as a submodule:

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

[ROOT](https://root.cern/install/) is a required dependency. There is a known conflict between AmpTools' GPU usage and RooFit/TMVA which comes with the conda-forge binaries of ROOT. Therefore, it is unfortunate, that ROOT has to be built from source with roofit and tmva off. See [Building ROOT from source](https://root.cern/install/build_from_source/) and add `-Droofit=OFF -Dtmva=OFF` as cmake arguments

<!-- ```shell
# mamba install -c conda-forge root=6.26 # ifarm nvcc requires gcc<11. 6.28 ships with 12
``` -->

Modify `set_environment.sh` to match you GPU environment (default: setup for JLab ifarm)

```shell
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d # load environment variables on conda activation
```

Build required libraries

```shell
# Options in brackets are optional
cd $REPO_HOME/external
make [mpi/gpu/mpigpu/gpumpi] # distributes modified Makefiles and makes AmpTools + AMPS_DATAIO into a shared libraries
```

Simple Unit tests

```shell
sed -i "s~REPLACE_FOLDER_LOCATION~$REPO_HOME/tests/samples/SIMPLE_EXAMPLE~" $REPO_HOME/tests/samples/SIMPLE_EXAMPLE/fit.cfg # update path
pytest -v
pytest -k [marked-test] # to run a specific marked test defined in pytest.ini
```
