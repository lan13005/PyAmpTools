# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools). Under the hood, it uses PyROOT which is based on cppyy. These bindings will
hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem.

Features:

- Access to PyROOT ecosystem
- Pythonization of c++ objects: simplify interactions with c++ source code, [additional information](https://root.cern/manual/python/#pythonizing-c-user-classes)
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
pip install mpi4py # MPI, mamba will link it against the wrong executables
mamba install -c conda-forge root=6.26 # ifarm nvcc requires gcc<11. 6.28 ships with 12
pre-commit install --install-hooks # (Optional) commit hooks to perform loose formatting
```

Modify `set_environment.sh` to match you GPU environment (default: setup for JLab ifarm)

```shell
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d # load environmnet variables on conda activation
```

Build required libraries

```shell
# Options in brackets are optional
cd $REPO_HOME/external
make [mpi/gpu/mpigpu/gpumpi] # distributes modified Makefiles and makes AmpTools + AMPS/DATAIO into a shared libraries
```

Simple Unit tests

```shell
sed -i "s~REPLACE_FOLDER_LOCATION~$REPO_HOME/tests~" $REPO_HOME/tests/samples/fit_res.cfg # update path
pytest -v
```
