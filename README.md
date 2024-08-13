# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools). Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. There are no known features of AmpTools that is not currently supported and both GPU and MPI is working. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

Features:

- Access to PyROOT ecosystem
- [Pythonization](https://root.cern/manual/python/#pythonizing-c-user-classes) of c++ objects: simplify interactions with c++ source code
- Dynamically load appropriate libraries / (re)compilation of high level scripts (like fits and plotters) are time consuming and distracting
- Python ecosystem:
  - Improved scripting, string parsing (regex)
  - Markov Chain Monte Carlo: [emcee](https://emcee.readthedocs.io/en/stable/)
  - ...

# Documentation

Here is the [Documentation](https://lan13005.github.io/PyAmpTools/intro.html) for installation instructions / api / and tutorials.
