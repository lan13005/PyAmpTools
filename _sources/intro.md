# PyAmpTools

Quantum Chromodynamics in the low energy regime is poorly understood. Gluon rich systems, like hybrid mesons, are predicted by theory but experimental idenficiation of these states have been difficult. The light meson spectrum contains many overlapping states that require partial wave (amplitude) analysis to disentangle. [AmpTools](https://github.com/mashephe/AmpTools) is a library written in c++ that facilitates performing unbinned maximum likelihood fits of experimental data to a coherent sum of amplitudes.

This repository contains Python bindings for AmpTools. Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

---

# Usage / Design

AmpTools and FSRoot are included as git submodules. Modified source files and makefiles are included in `external` directory to build a set of shared libraries that can then be imported into PyROOT.  Amplitude definitions and Data I/O are located in `external/AMPTOOLS_AMPS_DATAIO`. Additional amplitudes and data readers can be directly added to the folder and then re-maked. A variation of `gen_amp`, a program to produce simulations with AmpTools, is provided in `external/AMPTOOLS_GENERATORS`. This distribution system will unfortunately lag behind the sources from `halld_sim` and `AmpTools` repos.

`pa` is an **executable** that dispatches various functions for: maximimum likelihood fitting (`fit`), extracting fit fractions from MLE fit results (`fitfrac`), and Markov chain Monte Carlo (`mcmc`), and simulation generation (`gen_amp`, `gen_vec_ps`). Example usage:

```
pa -h # for usage
pa fit my_amptools.cfg # to run MLE fit
pa mcmc -h # for usage of mcmc
pa fitfrac amptools_results.fit # extract fit fractions
pa gen_amp my_amptools.cfg # to generate simulations
```

Additional files in the `scripts` folder are provided that perform `amptools` configuration file generation and plotting of the MLE fit results. These scripts are reaction dependent, therefore should be used as references to build upon. Utility functions are located in the `utility` folder.

---

# Installation

The following setup uses the `Bash` Shell and Anaconda. `conda` can be very slow to resolve dependencies for ROOT. Use [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) instead of conda for faster dependency resolution

```shell
# No need for recuse-submodules flag if using pre-compiled libraries at JLab
git clone https://github.com/lan13005/PyAmpTools --recurse-submodules
cd PyAmpTools
source set_environment.sh # detects if on JLab and loads some envs, else does nothing for now
```

All major dependencies (ROOT, AmpTools, FSRoot) are built from source.
[ROOT](https://root.cern/install/) >v6.26 is a required dependency (build steps are shown below) requiring at least some version of `gcc` (9.3.0 works).  See [Building External Libraries](#Sourcing) to build these libraries first. **Note: If you are running on the Jefferson Lab system the setup script will detect it and will load the default pre-compiled external libraries for a more streamlined experience.**

```shell
conda env create  # Creates environment specified by environment.yml and pyproject.toml
conda activate pyamptools # activate the environment
# pip install mpi4py if you wish to use MPI (see Potential Build Errors if failing)
pre-commit install --install-hooks # (Optional) commit hooks to perform loose formatting
```

Then create the necessary directory and link the main environment script, allowing for `set_environment.sh` to be sourced everytime `conda activate pyamptools` is executed. **Note:** VSCode loads the environment but does not appear to run `activate.d` and therefore requires manual activation.

```shell
# Modify set_environment.sh for your setup
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
source set_environment.sh # manually source for now
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d/ # setup auto-load for next time
```

## Building External Libraries (AmpTools, ROOT) <a id="Sourcing"></a>

If you are not on the JLab system or do not want to use the default pre-compiled libraries on the JLab file system then you are in the right section.

There is a known conflict between AmpTools' GPU usage and RooFit/TMVA which comes with the conda-forge binaries of ROOT. Currently, ROOT has to be built from source with roofit and tmva off. A build script is included to download ROOT from source with the appropriate cmake flags to achieve this. Please modify `set_environment.sh` to match you hardware (GPU, etc) environment

```shell
cd external/root
# Modify build_root.sh to match your environment (gcc versions, etc))
# if you modify the root version and use VSCode please update .vscode/settings.json file's extraPaths variable accordingly
source build_root.sh
cd ../.. # move back to main directory
source set_environment.sh # to load the new ROOT environment before downloading amptools/fsroot
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
sed -i "s~REPLACE_FOLDER_LOCATION~$PYAMPTOOLS_HOME/tests/samples/SIMPLE_EXAMPLE~" $PYAMPTOOLS_HOME/tests/samples/SIMPLE_EXAMPLE/fit.cfg # update path
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
