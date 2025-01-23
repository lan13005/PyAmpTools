# PyAmpTools

Quantum Chromodynamics in the low energy regime is poorly understood. Gluon rich systems, like hybrid mesons, are predicted by theory but experimental idenficiation of these states have been difficult. The light meson spectrum contains many overlapping states that require partial wave (amplitude) analysis to disentangle. [AmpTools](https://github.com/mashephe/AmpTools) is a library written in c++ that facilitates performing unbinned maximum likelihood fits of experimental data to a coherent sum of amplitudes.

This repository contains Python bindings for AmpTools. Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

---

# Usage / Design

AmpTools and FSRoot are included as git submodules. Modified source files and makefiles are included in `external` directory to build a set of shared libraries that can then be imported into PyROOT.  Amplitude definitions and Data I/O are located in `external/AMPTOOLS_AMPS_DATAIO`. Additional amplitudes and data readers can be directly added to the folder and then re-maked. A variation of `gen_amp`, a program to produce simulations with AmpTools, is provided in `external/AMPTOOLS_GENERATORS`. This distribution system will unfortunately lag behind the sources from `halld_sim` and `AmpTools` repos.

`pa` is an **executable** that dispatches various functions for: maximimum likelihood fitting (`fit`), extracting fit fractions from MLE fit results (`fitfrac`), and Markov chain Monte Carlo (`mcmc`), and simulation generation (`gen_amp`, `gen_vec_ps`). Example usage:

```
pa -h # for usage
pa -f # pa is a simple dispatch system, this command shows the file locations of each command
pa fit my_amptools.cfg # to run MLE fit
pa mcmc -h # for usage of mcmc
pa fitfrac amptools_results.fit # extract fit fractions
pa gen_amp my_amptools.cfg # to generate simulations
```

Additional files in the `scripts` folder are provided that perform `amptools` configuration file generation and plotting of the MLE fit results. These scripts are reaction dependent, therefore should be used as references to build upon. Utility functions are located in the `utility` folder.
    
---

# Installation

Please make modifications to `DockerInstall.sh` to match your environment. These installation steps builds on top of a minimal base image/environment. Some important considerations:
- if you already have conda then comment out those lines
- default is to build with mpi support
- if you need gpu support or gpu+mpi support then you can still go into `external` folder and run `make mpigpi` and additional shared libraries will be built

```shell
source DockerInstall.sh
```

## Build Docker Image (developers)

```shell
./DockerBuild.sh <GH_USERNAME> <GH_PAT> # github username and pat needed for now due to private repo access
```

# Apptainer Usage

```shell
# /scratch needed for MPI and fontconfig for matplotlib(not so important)
# NOTE: Do not bind ~
apptainer exec --contain \
    --bind /working/directory \
    --bind /data/directory \
    --bind /scratch \
    --bind ~/.cache/fontconfig \
    --env BASH_ENV=/dev/null \
    pyamptools.sif bash

# INSIDE THE CONTAINER:
source ~/.bashrc

pa -h # to see usage of pa
```

```{note}
`vscode` works inside containers with `Remote - Containers` extension. Unfortunately, it does not work `apptainer`. I followed [this Github solution using ssh](https://github.com/oschulz/container-env) to get it working.  
```

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
