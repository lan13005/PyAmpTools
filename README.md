# Introduction

This repository contains Python bindings for [AmpTools](https://github.com/mashephe/AmpTools) and interfaces with `iftpwa` repository for constructing non-parametric models for partial wave analysis under the numerical information field theory framework, [NIFTy](https://github.com/NIFTy-PPL/NIFTy). Additional hooks into jax + numpyro allows for rapid and automated exploration of the partial wave optimization landscape under Frequentist and Bayesian lenses.

## Information Field Theory for Partial Wave Analysis
- `iftpwa` repository (currently private under development)
- Partial wave analysis of large datasets using non-parametric (smooth Gaussian processes) and parametric models using [NIFTy](https://github.com/NIFTy-PPL/NIFTy) for fast variational inference over million to billion parameter spaces.
  - For general information on information theory, see these [Notes](https://lan13005.github.io/Information-Theory/)

## Automated Input/Output studies for partial wave analysis
- `iftpwa` prior model can act as a **generator** for diverse and complex amplitude models (smooth Gaussian processes + parametric models like a Breit-Wigner, Flatte, etc.)
- Inference from multiple angles using likelihoods provided by [JAX](https://jax.readthedocs.io/en/latest/index.html):
  - Maximum Likelihood Estimation (MLE) using [iminuit](https://iminuit.readthedocs.io/en/latest/index.html) or `scipy.optimize` (lbfgs, etc.)
  - MCMC (Hamiltonian Monte Carlo) using [NumPyro](https://num.pyro.ai/en/stable/index.html)
  - SVGD (Stein Variational Gradient Descent) using [NumPyro](https://num.pyro.ai/en/stable/index.html) for "inversion" of projected moments to amplitudes
- Neural density ratio estimation (DRE) for amortized application of acceptance functions - trained using flax/optax
- These core technologies can be used to increase the rate and effectiveness of Input/Output studies

## PyAmpTools Core

[AmpTools](https://github.com/mashephe/AmpTools) and [FSRoot](https://github.com/remitche66/FSRoot) are included as submodules. Python bindings are created using [PyROOT](https://root.cern/manual/python/) which uses [cppyy](https://cppyy.readthedocs.io/en/latest/index.html).

Features:

- Access to PyROOT ecosystem
- [Pythonization](https://root.cern/manual/python/#pythonizing-c-user-classes) of c++ objects: simplify interactions with c++ source code
- Dynamically load appropriate libraries / (re)compilation of high level scripts (like fits and plotters) are time consuming and distracting
- Improved scripting, string parsing (regex), etc.

# Documentation

Here is the [Documentation](https://lan13005.github.io/PyAmpTools/intro.html) for installation instructions and tutorials.
