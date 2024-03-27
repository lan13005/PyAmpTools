# PyAmpTools

Quantum Chromodynamics in the low energy regime is poorly understood. Gluon rich systems, like hybrid mesons, are predicted by theory but experimental idenficiation of these states have been difficult. The light meson spectrum contains many overlapping states that require partial wave (amplitude) analysis to disentangle. [AmpTools](https://github.com/mashephe/AmpTools) is a library written in c++ that facilitates performing unbinned maximum likelihood fits of experimental data to a coherent sum of amplitudes.

This repository contains Python bindings for AmpTools. Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

---

# Usage / Design

AmpTools and FSRoot are included as git submodules. Modified source files and makefiles are included in `external` directory to build a set of shared libraries that can then be imported into PyROOT.  Amplitude definitions and Data I/O are located in `external/AMPTOOLS_AMPS_DATAIO`. Additional amplitudes and data readers can be directly added to the folder and then re-maked. A variation of `gen_amp`, a program to produce simulations with AmpTools, is provided in `external/AMPTOOLS_GENERATORS` but is not built by the main makefile, a separate makefile is included with that directory.

Currently there exists **executables** for maximimum likelihood fitting (`amp_fit`), extracting fit fractions from MLE fit results (`amp_fitfrac`), and Markov chain Monte Carlo (`amp_mcmc`). Additional files in the `scripts` folder are provided that perform `amptools` configuration file generation and plotting of the MLE fit results. These scripts are reaction dependent, therefore should be used as references to build upon. Utility functions are located in the `utility` folder.

---

# Installation

The following setup uses the `Bash` Shell. All major dependencies (ROOT, AmpTools, FSRoot) are built from source.
[ROOT](https://root.cern/install/) >v6.26 is a required dependency (build steps are shown below) requiring at least some version of `gcc` (9.3.0 works). For JLab installations one should run the following bash commands or append to `.bashrc` file

```shell
# Append to bashrc file
source /etc/profile.d/modules.sh
module use /apps/modulefiles
module load mpi/openmpi3-x86_64
module load gcc/9.3.0
```

Install AmpTools and FSRoot as a submodule:

```shell
git clone https://github.com/lan13005/PyAmpTools --recurse-submodules
cd PyAmpTools
```

Environment setup. `conda` can be very slow to resolve dependencies for ROOT. Use [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) instead of conda for faster dependency resolution

```shell
conda env create  # Creates environment specified by environment.yml and pyproject.toml
conda activate pyamptools # activate the environment
pip install mpi4py # MPI (if available), mamba will link it against the wrong executables.
# if installing mpi4py fails, see bottom of page
pre-commit install --install-hooks # (Optional) commit hooks to perform loose formatting
```

There is a known conflict between AmpTools' GPU usage and RooFit/TMVA which comes with the conda-forge binaries of ROOT. Currently, ROOT has to be built from source with roofit and tmva off. A build script is included to download ROOT from source with the appropriate cmake flags to achieve this

```shell
cd external/root
# Modify build_root.sh to match your environment (gcc versions, etc))
# if you modify the root version and use VSCode please update .vscode/settings.json file's extraPaths variable accordingly
source build_root.sh
cd ../.. # move back to main directory
```

Modify `set_environment.sh` to match you GPU environment (default: setup for JLab ifarm). Then create the necessary directory and link the environment script, allowing for `set_environment.sh` to be sourced everytime `conda activate pyamptools` is executed. **Note:** VSCode loads the environment but does not appear to run `activate.d` and therefore requires manual activation.

```shell
# Modify set_environment.sh for your setup
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
source set_environment.sh # manually source for now
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d/ # setup auto-load for next time
```

Build required libraries

```shell
# Options in brackets are optional
cd external
make [mpi/gpu/mpigpu/gpumpi] # distributes modified Makefiles and makes libraries
```

---

# Additional Information

## Jupyter notebooks in VSCode

* Enter `jupyter-notebook --no-browser --port=8888` into the terminal
    * Copy localhost URL of the form: `http://localhost:8888/tree?token=e9aba1fab24532ceb89e29ba4485d8639ca4f4b41c490b91`
* Open your jupyter-notebook
    * Select Another Kernel
    * Connect to 'existing jupyter server'
    * Enter localhost url
    * Name it anything you like

# Unit Testing

```shell
sed -i "s~REPLACE_FOLDER_LOCATION~$REPO_HOME/tests/samples/SIMPLE_EXAMPLE~" $REPO_HOME/tests/samples/SIMPLE_EXAMPLE/fit.cfg # update path
pytest -v # -s to not hide stdout
pytest -k [marked-test] # to run a specific marked test defined in pytest.ini
```

# Building documentation

Documentation is powered by jupyter-book (a distribution of sphinx). A makefile is prepared to **build** and **clean** the documentation and **push** the changes to github pages. All three steps can be performed with the **rebuild** recipe.

To rebuild web documentation

```
cd docs
make rebuild # or choose one [clean/build/push/rebuild]
```
