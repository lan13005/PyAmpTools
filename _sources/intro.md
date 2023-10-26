# PyAmpTools

Quantum Chromodynamics in the low energy regime is poorly understood. Gluon rich systems, like hybrid mesons, are predicted by theory but experimental idenficiation of these states have been difficult. The light meson spectrum contains many overlapping states that require partial wave (amplitude) analysis to disentangle. [AmpTools](https://github.com/mashephe/AmpTools) is a library written in c++ that facilitates performing unbinned maximum likelihood fits of experimental data to a coherent sum of amplitudes. 

This repository contains Python bindings for AmpTools. Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.


# Usage / Design

AmpTools and FSRoot are included as git submodules. Modified source files and makefiles are included in `external` directory to build a set of shared libraries that can then be imported into PyROOT.  Amplitude definitions and Data I/O are located in `external/AMPTOOLS_AMPS_DATAIO`. Additional amplitudes and data readers can be directly added to the folder and then re-maked. A variation of `gen_amp`, a program to produce simulations with AmpTools, is provied in `external/AMPTOOLS_GENERATORS` but is not built by the main makefile, a separate makefile is included with that directory. 

Currently, the main scripts that perform an analysis, from simulation to fitting to plotting results, is located in `EXAMPLES/python` folder. These scripts can also be run from the commandline but its main functionality can be imported into another script (or Jupyter). Utility functions used by these scripts are located in the `utils` folder.

```{tableofcontents}
```
