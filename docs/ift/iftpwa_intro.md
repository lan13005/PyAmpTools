# iftpwa - Introduction

## Information Field Theory for Partial Wave Analysis (iftpwa)

When multiple resonances overlap in phase space, partial wave analysis is needed to disentangle the individual contributions. One can think of it as a generalized Fourier analysis where each basis component can represent a physically interpretable process (caveats!). 

```{seealso}
Want to learn more about Information Field Theory? Checkout these [Notes](https://lan13005.github.io/Information-Theory/)
```

## Design Requirements
- uncertainty quantification via a Bayesian framework
- handles high dimensionality via flexible prior models for regularization
- allow analysis of large datasets using MPI and variational inference
- complex valued parameters
- modular and extensible / collaboration agnostic

This led to the development of the Information Field Theory for Partial Wave Analysis (iftpwa) package by Florian Kaspar (et al.) at the Technical University of Munich (TUM) with support from the Max Planck Institute for Astrophysics (MPA), which develops the Numerical Information Field Theory (NIFTy) probabilistic programming framework for Bayesian inference. GlueX later joined in on the project which I am a part of as of this writing. GlueX uses a polarized photon beam which allows access to additional information on the reaction dynamics. This additional information comes at the cost of increased complexity in the analysis and can lead to unstable results using traditional methods which Bayesian methods like NIFTy could help regulate.

### References
- [Progress in the partial-wave analysis methods at COMPASS](https://arxiv.org/abs/2311.00449)
- [NIFTy gitlab](https://gitlab.mpcdf.mpg.de/ift/nifty/-/tree/NIFTy_8/src?ref_type=heads)
