# Introduction

## Default Configuration and Options

This repository attaches onto several optimization frameworks: iminuit, scipy.optimize, numpyro, JAX. This allows for studying of the same model with different optimization strategies with likelihoods, and therefore derivatives, by JAX. This allows users to choose between maximum likelihood estimation (MLE) with and without regularization, Markov Chain Monte Carlo (MCMC) with different samplers under the numpyro umbrella, and Information Field Theory (IFT) using the iftpwa + NIFTy repositories. These differ in the optimization strategy and therefore each has their own tradeoffs but primarily revolves around the general idea of regularization. 

The default configuration is set up for $\gamma p \rightarrow \eta \pi^0 p$ channel.

```shell
# Make a copy of the default configuration file and dump to main.yaml
pa from_default -o main.yaml
```

```yaml
# Expected data folder structure. Sharing accmc and genmc is optional. Bkgnd is optional.
accmc.root # shares this acceptance MC dataset for all polarizations
data000.root # data datasets for each polarization
data045.root # data datasets for each polarization
data090.root # data datasets for each polarization
data135.root # data datasets for each polarization
genmc.root # shares this generated/thrown MC dataset for all polarizations
bkgnd000.root # (optional) background MC datasets for each polarization
bkgnd045.root # (optional) background MC datasets for each polarization
bkgnd090.root # (optional) background MC datasets for each polarization
bkgnd135.root # (optional) background MC datasets for each polarization
```

### Vector Pseudoscalar Channels

The IFT model description remains the same and the default settings can work well. The only thing we have to consider is how to tell `PyAmpTools` how to generate the required `AmpTools` configuration file and of course use a partial wave naming scheme that it can understand. Below is an example of for the $\gamma p \rightarrow \omega\eta p \rightarrow \eta \pi^+ \pi^- \pi^0 p$ channel. 

Here we only highlight *some* specific keys that are different from the default

```yaml
waveset: 1Sp1-_1Sp1+_1Sp0-_1Sp0+_1Pp1-_1Pp1+_1Pp0-_1Pp0+_2Pp0+_2Pp0-_2Pp1+_2Pp1-_2Pp2+_2Pp2-_3Fp0+_3Fp0-_3Fp1+_3Fp1-_3Fp2+_3Fp2-_3Fp3+_3Fp3-
phase_reference: 1Pp1-_1Pp1+
reaction: Beam Proton Eta Pi0 Pi+ Pi-
datareader: ROOTDataReaderTEM .1 .6 8.2 8.8 1.325 2.0
add_amp_factor: OmegaDalitz 0.1212 0.0257 0.0 0.0
append_to_decay: omega3pi
```