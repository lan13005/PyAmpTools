# iftpwa - Introduction

## Information Field Theory for Partial Wave Analysis (iftpwa)

When multiple resonances overlap in phase space, partial wave analysis is needed to disentangle the individual contributions. One can think of it as a generalized Fourier analysis where each basis component can represent a physically interpretable process (caveats!). 

```{seealso}
Want to learn more about Information Field Theory? Checkout these [Notes](https://lan13005.github.io/Information-Theory/)
```

## Design Requirements
- complex valued
- large datasets
- model dependence and degrees of freedom
  - high dimensionality
  - smoothness
- modular and extensible / collaboration agnostic
- uncertainty quantification - Bayesian analysis
- fast and scalable

This led to the initial development of the Information Field Theory for Partial Wave Analysis (iftpwa) package by researchers at the Technical University of Munich (TUM) with support from the Max Planck Institute for Astrophysics (MPA), which develops the Numerical Information Field Theory (NIFTy) probabilistic programming framework for Bayesian inference. GlueX later joined in on the project.

### Optimization
NIFTy provides [Metric Gaussian Variational Inference](https://arxiv.org/abs/1901.11033) and [Geometric Variational Inference](https://arxiv.org/abs/2105.10470) methods for optimization. Both of these approaches alternate between KL optimization for a specific shape of the variational posterior and updating the shape of the variational posterior. A *global iteration* is complete when both optimization steps conclude. *geoVI* reshapes the posterior to be more Gausssian-like for more effective inference.

### References
- [Progress in the partial-wave analysis methods at COMPASS](https://arxiv.org/abs/2311.00449)
- [NIFTy gitlab](https://gitlab.mpcdf.mpg.de/ift/nifty/-/tree/NIFTy_8/src?ref_type=heads)
