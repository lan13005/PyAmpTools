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
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d # load environmnet variables on conda activation
pre-commit install --install-hooks # (Optional) commit hooks to perform loose formatting
```

Build required libraries

```shell
# Options in brackets are optional
cd $REPO_HOME/external/MakeAmpTools/
make [mpi] # distributes modified Makefiles and makes AmpTools into a shared library
cd $REPO_HOME/external/AMPTOOLS_AMPS; make [MPI=1]
cd $REPO_HOME/external/AMPTOOLS_DATAIO; make [MPI=1]
```

Simple Unit tests

```shell
pytest -v
```
