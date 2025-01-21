# iftpwa - Default Configuration

## Correlated Field Model in NIFTy
Please read the following first, perhaps run the code yourself to see how the correlated field model works in NIFTy. This model forms the basis of the iftpwa package.
[Showcasing the Correlated Field Model (NIFTy)](https://ift.pages.mpcdf.de/nifty/user/old_nifty_getting_started_4_CorrelatedFields.html)

```{note}
The correlated field model is no longer a Gaussian process but it does make understanding and explaining the IFT approach easier
```

## Simplified Model

The following simplified model is used to depict how the configuration file key-value fields affect the physics model:

$$
A_i & = \kappa \cdot \rho_i \cdot C_i \cdot \left[ G(w_i, s_i, f_i, a_i) \times I(w_i) + a_i P_i(w_i, \vec{x}) \right] \\
$$

$$
\begin{split}
i & = \text{wave index} \\ 
\kappa & = \text{kinematic factor} \\ 
\rho_i & = \text{phase space factor as a pkl file} \\
C_i & = \text{scale factor -> half-normal / laplace priors} \\
G(s_i, f_i, a_i) & = \text{Gaussian process with scale, flexibility, asperity} \\ 
I & = \text{indicator function to zero Gaussian process component} \\
a & = \text{prescale factor for Parametric component} \\
P(\vec{x}) & = \text{Parametric component with parameters } \vec{x} \\ 
\end{split}
$$

## Default Configuration for GlueX PWA

Below is the default YAML configuration for GlueX analyses of $\gamma p \rightarrow \eta\pi^0 p$

```yaml
GENERAL:
    pwa_manager: GLUEX
    seed: 42 # (int) RNG seed for reproducible results. Optimizer can get stuck in local minima so freely change this or try multiple values
    verbosityLevel: 1 # (int) Greater than 0 will print a bit (slightly) more information to screen
    outputFolder: ??? # (path) Base folder to dump all results into
    fitResultPath: ${GENERAL.outputFolder}/niftypwa_fit.pkl # (path) Python pickle file dump location, pkl file stores all the information about {input, model, result}
    maxNIFTyJobs: 1 # (int) Number of NIFTy jobs (not to be confused with MPI processes that the PWA_MANAGER can spawn). For example, if one would like to use multiple NIFTy processes for sampling. Untested.
    initial_position_std: 0.1 # (float) Random initial position is drawn from your prior model, Gaussian distributed with this standard deviation. See NIFTy/src/minimization/optimize_kl.py for more information
    # initial_position_template: /w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0/RESULTS_JAX_P/NiftyFits_a2B_a2pB_a0F/niftypwa_fit.pkl
    default_yaml: null # (path) values in this file can be used to update a yaml file found at the path specified with this key. Allows reusage.

# pwa_manager for GlueX takes in another configuration file that defines the partial waves used and the kinematic binning: (mass, t)
#   This maintains consistency between iftpwa results and results from a set of maximum likelihood fits obtained using AmpTools
PWA_MANAGER:
    yaml_file: /w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0/jax_primary_P.yaml

# Define our forward model containing correlated fields and parametric models
IFT_MODEL:
    scale: auto # (float, "auto") overal scale factor for amplitude
    nCalcScale: 500 # (float) number of samples used to estimate IFT_MODEL.scale
    positiveScale: half_normal # (half_normal, sqrt_exp) or (bool) choice of distribution using IFT_MODEL.scale. half_normal is a half normal, sqrt_exp is laplace (sqrt since we are on amplitude level not intensity). Bool is legacy to take absolute value of scale
    useLogNormalPriorSlope: false # (bool) use log normal prior for the slope of the power spectra instead of normal
    useGammaFlex: false # (bool) use Gamma distribution for the flexibility of the power spectra. Typically lighter tails than log-normal
    useLaplaceScale: false # (bool) should be same as setting positiveScale = sqrt_exp. We generally only consider positive scales so this is technically exponentially distributed
    useGammaAsp: false # (booL) use Gamma distribution for the asperity of the power spectra. Typically lighter tails than log-normal
    loglogavgslope: [-4.0, 0.1] # (float, float) for (mean, std) of the power spectrum slope (log-log scale). More negative slopes will limit high frequency components. Must be provided.
    flexibility: [0.5, 0.5] # (float, float) for (mean, std) or null. (0.5, 0.5) is special case resulting in Laplace distribution. Determines the amplitude of the integrated Wiener process component of the power spectrum.
    asperity: [0.5, 0.5] # (float, float) for (mean, std) or null. (0.5, 0.5) is special case resulting in Laplace distribution. Determines the roughness of the integrated Wiener process component of the power spectrum.
    stdOffset: [1.0, 0.001] # Amplitude offfset is normally distributed around 0 with this (float, float) for (mean, std). 
    loglogavgslopeTprime: [-4.0, 0.1] # (float, float). Cannot be null if you have multiple t-bins. 
    flexibilityTprime: [0.5, 0.5] # (float, float) or null. Read above
    asperityTprime: null # # (float, float) or null. Read above
    dofdexStye: real_imag # (real_imag, single_dofdex) allows fields to share the same power spectrum model. Useful for coupling real and imaginary parts of the same complex amplitude
    custom_model_path: ${PARAMETRIC_MODEL} # (path) or (YAML reference). Path must point to a file (see iftpwa/assets/parametric_model.py). (YAML reference) defined below
    modelName: null # (string) if custom_model_path is provided referencing a key in the model dictionary. null if custom_model_path is defined in this YAML file
    res2bkg: 1.0 # (float) scale parametric/resonance contributions before adding to bkgnd to form the signal
    bkg2res: 1.0 # (float) scale bkgnd contributions before adding to parametric/resonance to form the signal
    tPrimeFalloff: 0 # (float) preweight t distribution. Legacy
    estimateRatios: false # (bool) use the user-provided dataset ratios. Unused by GlueX
    perTprimeScaling: true # (bool) scale each t bin separately
    altTprimeNorm: no_norm # (peak_norm, no_norm, kin_norm) dictates how to normalize the phase space multiplier for each t-bin. no_norm keeps the normalization the supplied phase space pkl file
    noInterpScipy: false # (bool) use scipy.interpolate.interp1d cubic interpolation instead of nifty.interp linear interpolation to ensure values align in the phase space multiplier
    phaseSpaceMultiplier: null # (path) or null. Path to a pickled file containing a tuple of (masses, dict[amp: float]) where each float will scale the corresponding amp field
    s0: null # (float) or null. Energy scale factor. 
    productionFactor: GLUEX # (string) or null. Production factor scheme (see iftpwa/src/mode/model_builder.py). Depends on beam energy and target mass. 
    ratio_fluct: null # can use if using multiple datasets
    ratio_fluc_std: null # can use if using multiple datasets
    ratio_slope: null # can use if using multiple datasets
    ratio_stdSlope: null # can use if using multiple datasets
    ratio_slopeTprime: null # can use if using multiple datasets
    ratio_stdSlopeTprime: null # can use if using multiple datasets
    ratio_std_offset: null # can use if using multiple datasets 
    ratio_std_std_offset: null # can use if using multiple datasets

# Used in IFT_MODEL.custom_model_path
PARAMETRIC_MODEL:
    smoothScales: False # (bool) uses a correlated field to smooth the t-dependence of the parameteric models
    parameter_priors: # Define prior distributions for the free parameters 
        # see iftpwa/src/utilities/nifty.py for available priors but most parameters should be positive so log-normal is typical
        m_a0_980: UnnamedLogNormalOperator(sigma=0.05, mean=0.98) # Flatte peak mass for a0(980)
        g1_a0_980: UnnamedLogNormalOperator(sigma=0.1, mean=0.35) # Flatte coupling_1 for a0(980)
        g2_a0_980: UnnamedLogNormalOperator(sigma=0.1, mean=0.35) # Flatte coupling_2 for a0(980)
        m_a2_1320: UnnamedLogNormalOperator(sigma=0.0013 * 30, mean=1.3186) # Breit-Wigner peak mass for a2(1320)
        w_a2_1320: UnnamedLogNormalOperator(sigma=0.002 * 30, mean=0.105) # Breit-Wigner width for a2(1320)
        m_a2_1700: UnnamedLogNormalOperator(sigma=0.02, mean=1.706) # Breit-Wigner peak mass for a2(1700)
        w_a2_1700: UnnamedLogNormalOperator(sigma=0.05, mean=0.380) # Breit-Wigner width for a2(1700)
    resonances:
    - a0_980:
        name: "$a_0(980)$" # (string) name of the resonance in LaTeX format
        fun: "flatte" # (string) name of the function to use for the resonance. Available functions found at iftpwa/src/model/physics_functions.py
        preScale: 3.5 # preScale this component by this factor to bias towards total values. Can be 0
        no_bkg: false # (bool) if true, this component will not have a correlated field background component
        paras: {"mass": m_a0_980, "g1": g1_a0_980, "g2": g2_a0_980} # (dict) free parameters for the physics function
        static_paras: {"mass11": 0.548, "mass12": 0.135, "mass21": 0.495, "mass22": 0.495, channel: 1} # (dict) fixed parameters for the physics function
        waves: ['Sp0-', 'Sp0+'] # (List[string]) list of waves containing this parameteric component
    - a2_1320:
        name: "$a_2(1320)$"
        fun: "breitwigner_dyn"
        preScale: 1.5
        no_bkg: False
        paras: {"mass": m_a2_1320, "width": w_a2_1320}
        static_paras: {"spin": 2, "mass1": 0.548, "mass2": 0.135}
        waves: ['Dm2-', 'Dm1-', 'Dp0-', 'Dp1-', 'Dp2-', 'Dm2+', 'Dm1+', 'Dp0+', 'Dp1+', 'Dp2+']
    - a2_1700:
        name: "$a_2(1700)$"
        fun: "breitwigner_dyn"
        preScale: 5 # 0.25
        no_bkg: false # true
        paras: {"mass": m_a2_1700, "width": w_a2_1700}
        static_paras: {"spin": 2, "mass1": 0.548, "mass2": 0.135}
        waves: ['Dm2-', 'Dm1-', 'Dp0-', 'Dp1-', 'Dp2-', 'Dm2+', 'Dm1+', 'Dp0+', 'Dp1+', 'Dp2+']

# Some parameters takes a list of lists which dictate the number of global iterations in a given setting
#   See iftpwa/src/scripts/iftpwa_fit.py for calls to `makeCallableSimple` for parameters with this style
LIKELIHOOD:
    approximation: false # (bool) approximate the likelihood (with Gaussian, I believe)
    metric_type: normal # (normal, studentt) type of metric to use for the likelihood
    theta_studentt: null # (float) theta paraemeter for student-t distribution, this has to be set if metric_type is studentt
    learn_metric: false # (bool) can learn the metric but not recommended
    learn_metric_sigma: 1.0 # (float)
    nKeepMetric: 1 # (int) global iterations at nKeepMetric intervals will not approximate the likelihood. 1 means we never approximate the likelihood I guess
    # 
    subSampleProb: 1.0 # (float) subsample the data to speed up likelihood calculation
    initialSubSampleProb: 1.0 # (float) scales the initial signal field by this factor, in principle should match the initial subSampleProb
    subSampleApproxIndep: false # (bool) Legacy, unused
    bootstrap: false # (bool) bootstrap the data to introduce more stochasticity
    external_approximation: null # (path) pointing to an external pickle file, null if not used
    dropout_prob: 0.0 # (float) dropout probability for each kinematic bin. Introduces more stochasticity and can help with overfitting
    rotate_phase: null # (bool) or null. Unused
    clip_latent: 0 # (float) if between [0, 1] then this is a fractional scaling of the latent space between (-clip_latent, clip_latent) to prevent large updates. 

# Some parameters takes a list of lists which dictate the number of global iterations in a given setting
#   See iftpwa/src/scripts/iftpwa_fit.py for calls to `makeCallableSimple` for parameters with this style
OPTIMIZATION:
    nSamples: [[1, 0], [4, 5], [10, 10], 25] # (int) Runs 0 sampling (point estimate) for 1 global iteration, 5 samples for next 4 global iterations, ...
    nIterGlobal: 1 # (int) number of global iterations to run NIFTy optimize_kl for
    nMultiStart: 0 # (int) number of multiple restarts to perform (each only a fixed single global iteration) that optimizes multiStartObjective
    multiStartObjective: "maximize|Dp2+_fit_intens" # (str) pipe separated direction|quantity to select best starting condition. i.e. "minimize|energy" will choose the minimum energy (KL divergence/ELBO/Loss/Cost)
    
    # KL Minimizer
    algoOptKL: LBFGS # (string) algorithm
    nIterMaxOptKL: 50 # (int) number of iterations to perform algoOptKL to optimize the KL divergence
    deltaEKL: 0.001 # (float) if the difference between the last and current energies is below this value, the convergence counter will be increased in this iteration.
    convergenceLvlKL: 2 # (int) the number which the convergence counter must reach before the iteration is considered to be converged

    # Sampling
    # The accuracy of the samples specify the point at which they become dominated by noise. As to not constrain the KL minimization
    #    samples should be sampled more accurately than our KL optimization objective, hence, the more stringent minimization criterion for the sampling in the demonstration script
    #    See https://gitlab.mpcdf.mpg.de/ift/nifty/-/blob/NIFTy_8/demos/0_intro.py and https://github.com/NIFTy-PPL/NIFTy/issues/35
    nIterMaxSamples: 1000 # (int) iteration limit for sampling
    deltaESampling: 0.001 # (float) see above
    convergenceLvlSampling: 2 # (int) see above

    # GeoVI
    algoOptGeoVI: Newton # (string) algorithm
    nIterMaxGeoVI: 10 # (int) number of iterations to perform algoOptGeoVI
    deltaEGeoVI: 0.001 # (float) see above
    convergenceLvlGeoVI: 2 # (int) see above

    #############
    resume: true # (bool) resume previous fit or overwrite
    #############

    niftyCache: ${GENERAL.outputFolder}/ # (path) path to dump NIFTy results
    overwriteCache: true # (bool) overwrite the cache
    sanityChecks: false # (bool) NIFTy - some sanity checks that are evaluated at the beginning. They are potentially expensive because all likelihoods have to be instantiated multiple times
    constants: [tamps_mass_dir_loglogavgslope] # provide the names of the latent space variables that should be fixed ['tamps_mass_dir_loglogavgslope'] # examples: ['tamps_mass_dir_loglogavgslope', 'tamps_mass_dir_flexibility', 'tamps_tprime_dir_loglogavgslope', 'tamps_tprime_dir_flexibility', 'tamps_zeromode']

# Scans are performed by [Optuna](https://github.com/optuna/optuna). `iftpwa` naturally tracks the likelihood energy and the intensity of all individual amplitude components. 
#   If a ground truth is provided, the mean squared error or chi-squared (if number of samples > 1) are also calculated. 
#   These metrics are all available to be optimized against. These metrics take the form `{amplitude_field_name}_{"fit", resonance_name, "bkg"}_{"intens", "chi2"}`. 
#   A list of tracked metrics can be dumped at the end of a iftpwa fit if `verbosityLevel` > 0. 
#   Aggregated stats for `chi2` metric is also calculated: `{min, -2sigma, -1sigma, median, 1sigma, 2sigma, max}`. 
# Fields within this yaml can be referenced following a field path that is period delimited. 
#   An Optuna `suggest` command and associated args must also be passed. Field values must be a string that is then `eval`'d by `python` to construct `args` and `kwargs`. 
#   For example, `OPTIMIZATION.nSamples.1|suggest_int: "2, 10, step=2"` would modify the `nSamples` sub-field within the `OPTIMIZATION` field so that Optuna makes integer suggestions on list index element 1. The suggestions are integers between 2 and 10 in steps of 2. 
#   See Optuna docs for available `suggest` commands.
HYPERPARAMETER_SCANS:
    n_trials: 20
    objective: "minimize|energy" # (str) direction|objective where direction is either {'minimize', 'maximize'}
    sampler: RandomSampler # (str) see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html. BruteForceSampler can be used to exhaust all combinations
    GENERAL.seed|suggest_int: "0, 10000, step=1" # (str) randomizing the seed can be a proxy for running randomly initialized fits
    IFT_MODEL.loglogavgslope.1|suggest_float: "0.01, 0.51, step=0.25"
    IFT_MODEL.res2bkg|suggest_float: "0.01, 1.01, step=0.25"
    IFT_MODEL.scale|suggest_int: "1000, 21000, step=4000"
```

