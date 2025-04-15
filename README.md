# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools) and interfaces with `iftpwa` for constructing non-parametric models for partial wave analysis under the numerical information field theory framework, [NIFTy](https://github.com/NIFTy-PPL/NIFTy). Additional hooks into jax + numpyro allows for rapid and automated exploration of the partial wave optimization landscape under Frequentist and Bayesian lenses.

## Information Field Theory for Partial Wave Analysis (`iftpwa` companion repository)
Partial wave analysis of large datasets using non-parametric models using [NIFTy](https://github.com/NIFTy-PPL/NIFTy) for fast variational inference over million to billion parameter spaces.
- For general information on information theory, see these [Notes](https://lan13005.github.io/Information-Theory/)

## Automated Input/Output studies for partial wave analysis from numerous angles
- `iftpwa` prior model can act as a **generator** for complex amplitude models (smooth Gaussian processes + parametric models like a Breit-Wigner)
- Inference from multiple angles using likelihoods provided by [JAX](https://jax.readthedocs.io/en/latest/index.html):
  - Maximum Likelihood Estimation (MLE) using [iminuit](https://iminuit.readthedocs.io/en/latest/index.html) or `scipy.optimize` (lbfgs, etc.)
  - MCMC (Hamiltonian Monte Carlo) using [NumPyro](https://num.pyro.ai/en/stable/index.html)
  - SVGD (Stein Variational Gradient Descent) using [NumPyro](https://num.pyro.ai/en/stable/index.html) for "inversion" of projected moments to amplitudes
- Neural density ratio estimation (DRE) for amortized application of acceptance functions - trained using flax/optax
- These core technologies increases the effectiveness and rate at which Input/Output studies can be performed

## PyAmpTools Core
Under the hood, it uses [PyROOT](https://root.cern/manual/python/) which uses cppyy. These bindings will hopefully simplify the interaction with the AmpTools library while also providing access to the python ecosystem. There are no known features of AmpTools that is not currently supported and both GPU and MPI is working. [FSRoot](https://github.com/remitche66/FSRoot) is also included as a submodule and can be integrated into analysis workflows.

Features:

- Access to PyROOT ecosystem
- [Pythonization](https://root.cern/manual/python/#pythonizing-c-user-classes) of c++ objects: simplify interactions with c++ source code
- Dynamically load appropriate libraries / (re)compilation of high level scripts (like fits and plotters) are time consuming and distracting
- Improved scripting, string parsing (regex)

# Documentation

Here is the [Documentation](https://lan13005.github.io/PyAmpTools/intro.html) for installation instructions and tutorials.
