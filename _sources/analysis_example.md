# Analysis Chain Script

Once we have a pair of configuration files we are ready to perform our two types of fits and compare the results. `pa` is a simple dispatch system that calls various python scripts which each perform a specific task in this workflow.

This workflow goes from:
1. splits the raw data into kinematic bins
2. performs AmpTools MLE fits in each kinematic bin
3. performs IFT fit
4. draws tons of diagnostic plots and export fitted amplitudes / intensities / moments / etc into a file

```bash
set -e # Exit on error
set -o pipefail # Exit on error in pipeline

yaml=/LOCATION/TO/PYAMPTOOLS.YML
base_directory=$(grep "base_directory:" $yaml | awk '{print $2}') # grabs the base directory from the yaml file

echo "yaml: $yaml"
echo "base_directory: $base_directory"

pa run_cfgGen $yaml # Generate an AmpTools configuration file to be distributed 
pa run_divideData $yaml # Divide data into bins and distributes configuration files to each bin
pa run_mle $yaml # Perform all MLE fits across all bins and user specified randomly initialized fits
# pa calc_ps $yaml $ps_file # Create phase space factor file (pass to YAML field IFT_MODEL.phaseSpaceMultiplier) that multiples each partial wave (at the moment it is the barrier factor squared)
pa run_processEvents $yaml # Creates a data file (a representation of AmpVecs object in AmpTools) that GlueX iftpwa manager can use to reconstruct the likelihood for each bin

# Ideally we would like to be able to do some prior predictive checks to see what our unoptimized model looks like
# Unfortunately we have to `pa run_ift` before `iftPwaPriorPredict` since expanded yaml doesnt exist yet. This should be be fixed in the future.
# Example call: iftPwaPriorPredict --iftpwa_config $base_directory/NiftyFits/.nifty_expanded.yaml --plotFolder $base_directory/NiftyFits/prior_predictive

pa run_ift -v $yaml # Perform IFTPWA fit. If we wish to perform a hyperparameter search we can add the `--hyperopt` flag
rm -rf $base_directory/NiftyFits/plots # clean up plots from previous runs
# Draw diagnostic plots for the final fit results. Inside the output directory (see PyAmpTools YAML field: nifty.output_directory) is a diagnostics folder which draws the fit results on each global iteration
iftPwaPlot \
    --fitResult $base_directory/NiftyFits/niftypwa_fit.pkl \
    --plotFolder $base_directory/NiftyFits/plots \
    --n_prior_samples 1000 \
    --massIndepFit $base_directory/NiftyFits/.nifty_expanded.yaml
# Draws some convergence plots to see global iteration dependence of resonance parameters, distributions, etc
plotConvergence --inputFolder $base_directory/NiftyFits/fit_parameter_history --plotFolder $base_directory/NiftyFits/convergence_plots
pa run_momentPlotter $yaml -n 10 # make a subdirectory at NiftyFits/plots/moments overlaying projected moments from nifty and amptools fits
pa run_resultDump $yaml -n 10 # dump all results to csv files (if possible, it will calculate the moments also)
```