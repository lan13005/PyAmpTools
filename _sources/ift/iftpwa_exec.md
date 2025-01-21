# iftpwa - Default Execution

Once we have a pair of configuration files we are ready to perform our two types of fits and compare the results. `pa` is a simple dispatch system that calls various python scripts which each perform a specific task in this workflow.

```bash
yaml=/LOCATION/TO/PYAMPTOOLS.YML
subdir=$(grep "base_directory:" $yaml | awk '{print $2}') # grabs the base directory from the yaml file

pa run_cfgGen $yaml # Generate an AmpTools configuration file to be distributed 
pa run_divideData $yaml # Divide data into bins and distributes configuration files to each bin
pa run_mle $yaml # Perform all MLE fits across all bins and user specified randomly initialized fits
pa calc_ps $yaml $ps_file # Create phase space factor file that multiples each partial wave (at the moment it is the barrier factor squared)
pa run_processEvents $yaml # Creates a data file (a representation of AmpVecs object in AmpTools) that GlueX iftpwa manager can use to reconstruct the likelihood for each bin

# Ideally we would like to be able to do some prior predictive checks to see what our unoptimized model looks like
# Unfortunately we have to `pa run_ift` before `iftPwaPriorPredict` since expanded yaml doesnt exist yet. This should be be fixed in the future.
# Example call: iftPwaPriorPredict --iftpwa_config $subdir/NiftyFits/.nifty_expanded.yaml --plotFolder $subdir/NiftyFits/prior_predictive

pa run_ift $yaml # Perform IFTPWA fit. If we wish to perform a hyperparameter search we can add the `--hyperopt` flag
rm -rf $subdir/NiftyFits/plots # clean up plots from previous runs
iftPwaPlot \ # Draw diagnostic plots for the final fit results. Inside the output directory (see PyAmpTools YAML field: nifty.output_directory) is a diagnostics folder which draws the fit results on each global iteration
    --fitResult $subdir/NiftyFits/niftypwa_fit.pkl \
    --plotFolder $subdir/NiftyFits/plots \
    --n_prior_samples 1000 \
    --massIndepFit $subdir/NiftyFits/.nifty_expanded.yaml
plotConvergence --inputFolder $subdir/NiftyFits/fit_parameter_history --plotFolder $subdir/NiftyFits/convergence_plots # Draws some convergence plots to see global iteration dependence of resonance parameters, distributions, etc
```