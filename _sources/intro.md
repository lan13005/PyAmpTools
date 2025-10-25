# PyAmpTools

Quantum Chromodynamics in the low energy regime is poorly understood. Gluon rich systems, like hybrid mesons, are predicted by theory but experimental idenficiation of these states have been difficult. The light meson spectrum contains many overlapping states that require partial wave (amplitude) analysis to disentangle. This repository contains a collection of tools to perform partial wave analysis from Maximum Likelihood Estimation (MLE) to Markov Chain Monte Carlo (MCMC) to Information Field Theory (IFT) using likelihoods and gradients provided by [JAX](https://github.com/google/jax).

Additionally, this repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools) which is a library written in c++ that facilitates performing unbinned maximum likelihood fits of experimental data to a coherent sum of amplitudes. Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

---

# Usage / Design

`pa` is an **executable** that dispatches various functions to perform various tasks related to optimization and provides easy access to specific AmpTools functionality like generating simulations from `gen_amp` and `gen_vec_ps`, fitting, and extracting fit fractions. 

There are a variety of auxillary CLI programs that can perform filtering/selection, subsetting, and plotting of flat root files that are typically used in analysis workflows. Since this is a simple dispatch system, a user can easily add new CLI programs by placing them in the `bin` directory and adding their location to the `src/pyamptools/pa.py` script. This extensibility allows for a centralized hub of available tools to handle most analysis workflows. 

```shell
pa -h # for usage
pa -f # for dispatched file locations
```

AmpTools and FSRoot are included as git submodules. Modified source files and makefiles are included in `external` directory to build a set of shared libraries that can then be imported into PyROOT.  Amplitude definitions and Data I/O are located in `external/AMPTOOLS_AMPS_DATAIO`. Additional amplitudes and data readers can be directly added to the folder and then re-maked. A variation of `gen_amp`, a program to produce simulations with AmpTools, is provided in `external/AMPTOOLS_GENERATORS`. This distribution system will unfortunately lag behind the sources from `halld_sim`, `AmpTools`, and `FSRoot`.

---

# Installation

**Note:** Skip to *Apptainer Usage*, below, to simply use the container. The following instructions are for developers to build/maintain the container.

Please make modifications to `DockerInstall.sh` to match your environment. These installation steps builds on top of a minimal base image/environment. Some important considerations:
- if you already have conda then comment out those lines
- default is to build with mpi support
- if you need gpu support or gpu+mpi support then you can still go into `external` folder and run `make mpigpi` and additional shared libraries will be built
- simply call `source DockerInstall.sh` to install necessary dependencies (i.e. if you are in a minimal `almalinux` container environment already)
- instead, if you want to build the docker image simply call:

```shell
./DockerBuild.sh # ensure GH_USERNAME and GH_PAT are environment variables
```

## Apptainer Usage

```shell
# You might want to add `--cleanenv` flag to run the container in a clean environment
# /scratch needed for MPI and fontconfig for matplotlib(not so important)
# --writable-tmpfs allows temp modifications to sif contents. Ideally you can git pull and push updates changes to repos
# --bind whatever directories you need access to
apptainer exec --contain --writable-tmpfs \
    --bind /tmp \
    --bind /var/tmp \
    --bind /my/working/directory \
    --bind /my/data/directory \
    --bind /scratch \
    --bind ~/.cache/fontconfig \
    --env TMPDIR=/tmp,TEMP=/tmp,TMP=/tmp,BASH_ENV=/dev/null \
    /LOCATION/OF/pyamptools.sif bash
# Location on the jlab farm: /w/halld-scshelf2101/lng/WORK/PyAmpTools/pyamptools.sif

# INSIDE THE CONTAINER:
source /etc/bash.bashrc
```

### For VSCode Remote Containter Development

`vscode` remote development of containers is possible following [this Github solution using ssh](https://github.com/oschulz/container-env) to get it working. Install `cenv`, then call `cenv --help` for info to setup environment.

**NOTE:** This repo does not fully work when running remote container on macOS (arm) system since pypi wheels for jaxlib uses AVX hardware instruction set. 

```shell
# Example ssh config for vscode remote container on the jlab farm with proxyjump
#    using example an existing 'pyamptools' cenv
Host pyamptools~ifarm
  HostName ifarm.jlab.org
  ProxyJump login.jlab.org
  # ${HOME} in the container is actually /root which is not writable so mount local writable folder over it
  RemoteCommand bash --noprofile --norc -c "export CENV_APPTAINER_OPTS='--contain --writable-tmpfs -B /w/halld-scshelf2101/lng,/scratch'; ${HOME}/.local/bin/cenv pyamptools"
  RequestTTY yes
  user lng
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
