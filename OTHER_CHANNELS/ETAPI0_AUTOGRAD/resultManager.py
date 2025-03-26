from pyamptools.utility.general import load_yaml, calculate_subplot_grid_size, prettyLabels # TODO: must be loaded before loadIFTResultsFromPkl, IDKY yet
from pyamptools.utility.IO import loadIFTResultsFromPkl
from pyamptools.utility.IO import get_nEventsInBin
import pandas as pd
import numpy as np
import pickle as pkl
import glob
from typing import Tuple
import matplotlib.pyplot as plt
from rich.console import Console
import re
import os
import tqdm
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects
import mplhep as hep

# TODO:
# - Move MCMC and COMPARISONS to RESULTS_DATA, renaming COMPARISONS TO MLE (Keep AmpToolsFits as corroboration)
# - Move GENERATED to RESULTS_MC - prior sim function needs to be created to do this
# - MLE
#   - MLE folder should store everything. the pkl file should contain a name that is a suffix or tag that identifies it
#   - move optim_test to be a real function in pa.py
# - MCMC
#   - path_sampling and mcmc_test has to be made real functions in pa.py
#   - mcmc sampling should use total_intensity instead of total - go back and make column naming agree
# - symlink all plot subdirectories to RESULTS folder? (i.e. nifty results)
# - implement t-dependence?
# - This function should supercede run_resultsDump. ResultsManager can already do this, just move output helpful comments into the manager
# - add guards to not plot things that do not exist in the plotting functions

console = Console()

binNum_regex = re.compile(r'bin(\d+)')
header_fmt = "[bold blue]***************************************************\n{}\n[/bold blue]"

mle_color = 'xkcd:orange'
ift_color_dict = {
    "Signal": "xkcd:sea blue",
    "Param.": "xkcd:cerise",
    "Bkgnd": "xkcd:dark sea green"
}
gen_color = 'xkcd:green'
mcmc_color = 'black'

ift_alpha = 0.6 # transparency of IFT fit results
mle_jitter_scale = 0.1 # percent of the bin width the MLE results are jittered

# All plots are stored in this directory within the `base_directory` key in the YAML file
default_plot_subdir = "PLOTS"

class ResultManager:
    
    def __init__(self, yaml_file):
        
        self._mle_results  = pd.DataFrame()
        self._gen_results  = (pd.DataFrame(), pd.DataFrame()) # (generated samples of amplitudes, generated samples of resonance parameters)
        self._ift_results  = (pd.DataFrame(), pd.DataFrame()) # (fitted samples of amplitudes, fitted samples of resonance parameters)
        self._mcmc_results = pd.DataFrame()
        self._hist_results = pd.DataFrame()
        
        self.yaml_file = yaml_file
        self.yaml = load_yaml(yaml_file)
        self.base_directory = self.yaml['base_directory']
        self.waveNames = self.yaml['waveset'].split("_")
        self.phase_reference = self.yaml['phase_reference'].split("_")
        min_mass = self.yaml['min_mass']
        max_mass = self.yaml['max_mass']
        self.n_mass_bins = self.yaml['n_mass_bins']
        self.masses = np.linspace(min_mass, max_mass, self.n_mass_bins+1)
        self.mass_centers = (self.masses[:-1] + self.masses[1:]) / 2
        self.n_decimal_places = 5
        self.mass_centers = np.round(self.mass_centers, self.n_decimal_places)
        
        # Keep track of all the unique fit sources that the user loads
        #   Why would the user want to load multiple gen/hist sources?
        self.source_list = {
            'mle': [],
            'mcmc': [],
            'ift': [],
            'gen': [],
            'hist': []
        }
        
        # Identify incoherent sectors and create list of waves in each sector
        if not all([ref in self.waveNames for ref in self.phase_reference]):
            raise ValueError("Phase reference list contains waves that are not in the wave names list. Ensure reference waves are in the waveset!")
        self.sectors = {}
        self.sector_to_ref_wave = {}
        for ref_wave in self.phase_reference:
            self.sectors[ref_wave[-1]] = []
            self.sector_to_ref_wave[ref_wave[-1]] = f"{ref_wave}_amp"
        for wave in self.waveNames:
            self.sectors[wave[-1]].append(f"{wave}_amp")
        
        console.print(f"\n")
        console.print(header_fmt.format(f"Parsing {yaml_file} with these expected settings:"))
        console.print(f"wave_names: {self.waveNames}")
        console.print(f"identified {len(self.sectors)} incoherent sectors: {self.sectors}")
        console.print(f"n_mass_bins: {self.n_mass_bins}")
        console.print(f"mass_centers: {self.mass_centers}")
        console.print(f"\n")
        
    def attempt_load_all(self):
        if self._hist_results.empty:
            self._hist_results = self.load_hist_results()
        if self._mle_results.empty:
            self._mle_results = self.load_mle_results()
        if self._mcmc_results.empty:
            self._mcmc_results = self.load_mcmc_results()
        if self._ift_results[0].empty:
            self._ift_results = self.load_ift_results(source_type="FITTED")
        if self._gen_results[0].empty:
            self._gen_results = self.load_ift_results(source_type="GENERATED")

    @property
    def mle_results(self) -> pd.DataFrame:
        return self._mle_results
    
    @property
    def mcmc_results(self) -> pd.DataFrame:
        return self._mcmc_results
    
    @property
    def ift_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._ift_results
    
    @property
    def gen_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._gen_results
    
    @property
    def hist_results(self) -> pd.DataFrame:
        return self._hist_results
    
    def load_hist_results(self, base_directory=None, alias=None):
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
            
        source_name = self._check_and_get_source_name(self._hist_results, 'hist', base_directory, alias)
        if source_name is None:
            return self._hist_results
        
        console.print(header_fmt.format(f"Loading HISTOGRAM result from {base_directory}/AmpToolsFits"))
        nEvents, nEvents_err = get_nEventsInBin(f"{base_directory}/AmpToolsFits")
        hist_df = {'mass': self.mass_centers, 'nEvents': nEvents, 'nEvents_err': nEvents_err}
        hist_df = pd.DataFrame(hist_df)
        
        console.print(f"[bold green]Total Events Histogram Summary:[/bold green]")
        console.print(f"columns: {hist_df.columns}")
        console.print(f"shape: {hist_df.shape}")
        console.print(f"\n")
        
        hist_df = self._add_source_to_dataframe(hist_df, 'hist', source_name)
        if not self._hist_results.empty:
            console.print(f"[bold green]Previous histogram results were loaded already, concatenating[/bold green]")
            hist_df = pd.concat([self._hist_results, hist_df])
        self._hist_results = hist_df
        return self._hist_results
    
    def load_ift_results(self, base_directory=None, source_type="GENERATED", alias=None):
        
        """Currently only supports generated curves from iftpwa"""
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
            
        if source_type not in ["GENERATED", "FITTED"]:
            raise ValueError(f"Source type {source_type} not supported. Must be one of: [GENERATED, FITTED]")
        
        console.print(header_fmt.format(f"Loading {source_type} results from {base_directory}/{source_type}"))
        
        source_name = self._check_and_get_source_name(self._ift_results, 'ift', base_directory, alias)
        if source_name is None:
            return self._ift_results
        
        subdir = "GENERATED" if source_type == "GENERATED" else "NiftyFits"
        truth_loc = f"{self.base_directory}/{subdir}/niftypwa_fit.pkl"
        with open(truth_loc, "rb") as f:
            truth_pkl = pkl.load(f)
        ift_df, ift_res_df = loadIFTResultsFromPkl(truth_pkl)

        # requires loading hist results first to scale generated curves to data scale
        if source_type == "GENERATED":
            self._hist_results = self.load_hist_results()
            if self._hist_results is None:
                raise ValueError("Histogram results not found. We use this dataset to scale the generated curves, cannot proceed without it.")
            mle_totals = self._hist_results['nEvents'].sum()
            rescaling = mle_totals / np.sum(ift_df['fitted_intensity'])
            amp_cols = [c for c in ift_df.columns if c.endswith('_amp')]
            intensity_cols = [c for c in ift_df.columns if 'intensity' in c] + [amp.strip('_amp') for amp in amp_cols]
            ift_df[amp_cols] *= np.sqrt(rescaling)
            ift_df[intensity_cols] *= rescaling
        
        if len(ift_df) == 0:
            console.print(f"[bold red]No '{subdir.lower()}' curves found in {truth_loc}\n[/bold red]")
            return None, None
        
        # rotate away reference waves
        ift_df = self._rotate_away_reference_waves(ift_df)
        ift_df['mass'] = np.round(ift_df['mass'], self.n_decimal_places)
        
        ift_df = self._add_source_to_dataframe(ift_df, 'ift', source_name)
        ift_res_df = self._add_source_to_dataframe(ift_res_df, 'ift', source_name, append_to_source_list=False) # already appeneded above
        if not self._ift_results[0].empty:
            console.print(f"[bold green]Previous IFT results were loaded already, concatenating[/bold green]")
            ift_df = pd.concat([self._ift_results[0], ift_df])        
            ift_res_df = pd.concat([self._ift_results[1], ift_res_df])
        self._ift_results = (ift_df, ift_res_df)
        
        # NOTE: These print statements should come before self.hist_results is called since additional results will be loaded and printed
        console.print(f"[bold green]\nIFT {subdir} Summary:[/bold green]")
        console.print(f"Amplitudes DataFrame columns: {list(ift_df.columns)}")
        console.print(f"Amplitudes DataFrame shape: {ift_df.shape}")
        console.print(f"Resonance Parameters DataFrame columns: {list(ift_res_df.columns)}")
        console.print(f"Resonance Parameters DataFrame shape: {ift_res_df.shape}")
        console.print(f"\n")
        
        return self._ift_results

    def load_mle_results(self, base_directory=None, alias=None):
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
        
        console.print(header_fmt.format(f"Loading MLE results from {base_directory}"))
        
        source_name = self._check_and_get_source_name(self._mle_results, 'mle', base_directory, alias)
        if source_name is None:
            return self._mle_results

        def calculate_relative_phase_and_error(amp1, amp2, covariance):
            """
            This is created for MLE fits where we want to error propagate.
            The other approaches are based on Bayesian sampling so this is not needed
            
            Args:
                amp1: array, flattened array of real and imaginary parts of amplitude 1
                amp2: array, flattened array of real and imaginary parts of amplitude 2
                covariance: array, covariance matrix of the above amplitudes
            Returns:
                tuple: (relative phase, error)
            """
            
            # Calculate derivatives of relative phases with respect to parameters
            p_deriv = np.zeros(4)
            p_deriv[0] = -np.imag(amp1) / np.abs(amp1)**2  # d(phase)/d(a1_re)
            p_deriv[1] =  np.real(amp1) / np.abs(amp1)**2  # d(phase)/d(a1_im)
            p_deriv[2] =  np.imag(amp2) / np.abs(amp2)**2  # d(phase)/d(a2_re)
            p_deriv[3] = -np.real(amp2) / np.abs(amp2)**2  # d(phase)/d(a2_im)
            
            # Get indices in the covariance matrix
            variance = 0.0
            for i in range(4):
                for j in range(4):
                    variance += p_deriv[i] * p_deriv[j] * covariance[i, j]
                    
            amp1 *= np.exp(-1j * np.angle(amp2))
            relative_phase = np.angle(amp1, deg=True)
            relative_phase_error = np.sqrt(variance) * 180.0 / np.pi

            return relative_phase, relative_phase_error

        def load_from_pkl_list(pkl_list, mass_centers, waveNames):
            
            pkl_list = sorted(pkl_list, key=lambda x: int(binNum_regex.search(x).group(1)))

            results = {}
                        
            for pkl_file in pkl_list:
                with open(pkl_file, "rb") as f:
                    datas = pkl.load(f)
                    
                console.print(f"Loaded {len(datas)} fits with random starts from {pkl_file}")
                    
                bin_idx = int(pkl_file.split("_")[-2].lstrip("bin"))
                setting = int(pkl_file.split("_")[-1].rstrip(".pkl").lstrip("setting"))

                for i, data in enumerate(datas):

                    results.setdefault("setting", []).append(setting)                    
                    results.setdefault("bin", []).append(bin_idx)
                    results.setdefault("initial_likelihood", []).append(data["initial_likelihood"])
                    results.setdefault("likelihood", []).append(data["likelihood"])
                    results.setdefault("iteration", []).append(i)
                    
                    for key in data["final_par_values"].keys():
                        results.setdefault(key, []).append(data["final_par_values"][key])
                    
                    results.setdefault("total_intensity", []).append(data["total_intensity"])
                    results.setdefault("total_intensity_error", []).append(data["total_intensity_error"])
                    for waveName in waveNames:
                        results.setdefault(f"{waveName}", []).append(data[f"{waveName}"])
                        results.setdefault(f"{waveName}_error", []).append(data[f"{waveName}_error"])
                    
                    method = "tikhonov"
                    for iw, wave in enumerate(waveNames):
                        tag = f"{wave}_{method}"
                        covariance = data['covariances'][method]
                        reference_wave = self.sector_to_ref_wave[wave[-1]]
                        if covariance is None: # this can happen for Minuit if it fails to converge (IDK about scipy)
                            results.setdefault(f'{tag}_re_err', []).append(None)
                            results.setdefault(f'{tag}_im_err', []).append(None)
                            results.setdefault(f'{tag}_err_angle', []).append(None)
                        else:
                            assert covariance.shape == (2 * len(waveNames), 2 * len(waveNames))
                            cov_re = covariance[2*iw  , 2*iw  ]
                            cov_im = covariance[2*iw+1, 2*iw+1]
                            if np.isclose(cov_re, cov_im):
                                angle = 0.0
                            else:
                                rho = covariance[2*iw+1, 2*iw]  # off-diagonal term
                                angle = 0.5 * np.arctan2(2 * rho, cov_re - cov_im)
                            re_err = np.sqrt(max(0, cov_re))
                            im_err = np.sqrt(max(0, cov_im))
                            results.setdefault(f'{tag}_re_err', []).append(re_err)
                            results.setdefault(f'{tag}_im_err', []).append(im_err)
                            results.setdefault(f'{tag}_err_angle', []).append(angle)
                            reference_wave = self.sector_to_ref_wave[wave[-1]].strip("_amp")
                            amp1 = results[f"{wave}_amp"][-1] # get most recent value
                            amp2 = results[f"{reference_wave}_amp"][-1]
                            jw = waveNames.index(reference_wave)
                            # construct 4x4 submatrix of real imaginary parts for both amp1 and amp2
                            indicies = [2*iw, 2*iw+1, 2*jw, 2*jw+1]
                            submatrix = covariance[np.ix_(indicies, indicies)]
                            relative_phase, relative_phase_error = calculate_relative_phase_and_error(amp1, amp2, submatrix)
                            results.setdefault(f'{wave}_relative_phase', []).append(relative_phase)
                            results.setdefault(f'{wave}_relative_phase_error', []).append(relative_phase_error)

                    # for key in list(data["initial_guess_dict"].keys()) + ["total_intensity"]:
                    #     if key not in results: results[key] = [data[key]]
                    #     else: results[key].append(data[key])
                    
            results = pd.DataFrame(results)
            
            if len(results) == 0:
                console.print(f"[bold red]No results found in pkl files for: {pkl_list}\n[/bold red]")
                return None
            
            results['mass'] = results['bin'].apply(lambda x: mass_centers[x])
            
            results = results.sort_values(by=["mass", "setting"]).reset_index(drop=True)
            
            # rotate away reference waves
            results = self._rotate_away_reference_waves(results)
            return results
        
        pkl_list = glob.glob(f"{base_directory}/MLE/*pkl")
        mle_results = load_from_pkl_list(pkl_list, self.mass_centers, self.waveNames)
        
        if mle_results is None or len(mle_results) == 0:
            console.print(f"[bold red]No MLE results found in {base_directory}/MLE\n[/bold red]")
            return self._mle_results
        
        console.print(f"[bold green]\nMLE Summary:[/bold green]")
        console.print(f"columns: {list(mle_results.columns)}")
        console.print(f"n_random_starts: {mle_results.shape[0] // self.n_mass_bins}")
        console.print(f"shape: {mle_results.shape} ~ (n_bins * n_random_starts, columns)")        
        console.print(f"\n")
        
        mle_results['mass'] = np.round(mle_results['mass'], self.n_decimal_places)

        mle_results = self._add_source_to_dataframe(mle_results, 'mle', source_name)
        if not self._mle_results.empty:
            console.print(f"[bold green]Previous MLE results were loaded already, concatenating[/bold green]")
            mle_results = pd.concat([self._mle_results, mle_results])        
        self._mle_results = mle_results
        return self._mle_results
        
    def load_mcmc_results(self, base_directory=None, alias=None):
        """
        Loads MCMC results from a given base_directory
        
        Args:
            base_directory: str, path to the base directory containing 'MCMC' subdirectory. 
                If None, will load from self.base_directory specified in the yaml file.
            
        Returns:
            pd.DataFrame, MCMC results
        """
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
        
        console.print(header_fmt.format(f"Loading MCMC results from {base_directory}"))
        
        source_name = self._check_and_get_source_name(self._mcmc_results, 'mcmc', base_directory, alias)
        if source_name is None:
            return self._mcmc_results
        
        pkl_list = glob.glob(f"{base_directory}/MCMC/*_samples.pkl")
        if len(pkl_list) == 0:
            console.print(f"[bold red]No MCMC results found in {base_directory}/MCMC\n[/bold red]")
            return
        with open(pkl_list[0], "rb") as f:
            mcmc_results = pkl.load(f)
        mcmc_results = pd.DataFrame(mcmc_results)
        
        if len(mcmc_results) == 0:
            console.print(f"[bold red]No MCMC results found in {base_directory}/MCMC\n[/bold red]")
            return
        
        # rotate away reference waves
        mcmc_results = self._rotate_away_reference_waves(mcmc_results)
        mcmc_results['mass'] = np.round(mcmc_results['mass'], self.n_decimal_places)
        
        mcmc_results = self._add_source_to_dataframe(mcmc_results, 'mcmc', source_name)
        if not self._mcmc_results.empty:
            console.print(f"[bold green]Previous MCMC results were loaded already, concatenating[/bold green]")
            mcmc_results = pd.concat([self._mcmc_results, mcmc_results])
        self._mcmc_results = mcmc_results
        
        # print diagnostics
        console.print(f"[bold green]\nMCMC Summary:[/bold green]")
        console.print(f"columns: {list(self._mcmc_results.columns)}")
        console.print(f"n_samples: {self._mcmc_results.shape[0] // self.n_mass_bins}")
        console.print(f"shape: {self._mcmc_results.shape} ~ (n_samples * n_bins, columns)")
        console.print(f"\n")

        return self._mcmc_results
    
    def _rotate_away_reference_waves(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame is not a pandas DataFrame!")
        for sector in self.sectors:
            waves_in_sector = self.sectors[sector]
            df[waves_in_sector] *= np.exp(-1j * np.angle(df[self.sector_to_ref_wave[sector]]))[:, None] # broadcast this rotation to all waves in the sector
        return df
    
    def _check_and_get_source_name(self, df, source_type, base_directory, alias=None):
        # Check if DataFrame is already populated
        df_already_loaded = not (df is None or (isinstance(df, pd.DataFrame) and df.empty) or 
                               (isinstance(df, tuple) and (df[0] is None or df[0].empty)))

        if alias is None:
            # Default source is 'yaml' when loading from default base_directory in the yaml file
            if base_directory == self.base_directory or os.path.samefile(base_directory, self.base_directory):
                source_name = 'yaml'
                # Check if default source is already loaded
                if 'yaml' in self.source_list[source_type] and df_already_loaded:
                    console.print(f"[bold green]Default {source_type} results already loaded. Will return existing DataFrame[/bold green]")
                    source_name = None
                    return source_name
            else:
                raise ValueError(f"[bold red]Alias must be provided for the user-specified base_directory: {base_directory}\n[/bold red]")
        else:
            if alias in self.source_list[source_type]:
                raise ValueError(f"[bold red]{source_type} result with alias {alias} has already been loaded. Please choose a unique alias.[/bold red]")
            source_name = alias
        return source_name

    def _add_source_to_dataframe(self, df, source_type, source_name, append_to_source_list=True):
        if df is not None and not df.empty and source_name is not None:
            df['source'] = [source_name] * len(df)
        if append_to_source_list and source_name is not None:
            self.source_list[source_type].append(source_name)
        return df
    
def save_plot(output_file_location, fig, axes, overwrite=False, verbose=True):
    directory = os.path.dirname(output_file_location)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not overwrite and os.path.exists(output_file_location):
        if verbose: console.print(f"[bold red]File {output_file_location} already exists. Set overwrite=True to overwrite.\n[/bold red]")
        return

    for ax in axes.flatten(): # Detect unused axes and set them to not visible
        if not ax.has_data():
            ax.set_visible(False)

    fig.savefig(output_file_location)
    if verbose: console.print(f"[bold green]Creating plot at {output_file_location}[/bold green]")
    
def plot_gen_curves(resultManager: ResultManager, figsize=(10, 10)):
    
    ift_gen_df = resultManager.gen_results[0]
    
    console.print(f"\n[bold blue]Plotting 'generated curves' plots...[/bold blue]")
    
    waveNames = resultManager.waveNames
    mass_centers = resultManager.mass_centers
        
    nrows, ncols = calculate_subplot_grid_size(len(waveNames))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    for i, wave in enumerate(waveNames):
        irow = i // ncols
        icol = i % ncols
        wave_fit_fraction = ift_gen_df[wave] / ift_gen_df['fitted_intensity']
        axes[irow, icol].plot(mass_centers, wave_fit_fraction, color='black')
        axes[irow, icol].set_ylim(0, 1.1) # fit fractions, include a bit of buffer
        axes[irow, icol].set_xlim(mass_centers[0], mass_centers[-1])
        axes[irow, icol].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[irow, icol].tick_params(axis='both', which='major', labelsize=13)
        axes[irow, icol].text(0.2, 0.975, prettyLabels[wave],
                    size=20, color='black', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top',
                    transform=axes[irow, icol].transAxes)
        
    for i in range(ncols):
        axes[nrows-1, i].set_xlabel(r"$m_X$", size=15)
    for i in range(nrows):
        axes[i, 0].set_ylabel("Fit Fraction", size=15)

    plt.tight_layout()
    ofile = f"{resultManager.base_directory}/{default_plot_subdir}/gen_curves.png"
    save_plot(ofile, fig, axes, overwrite=True)
    plt.close()
    
def plot_binned_intensities(resultManager: ResultManager, bins_to_plot=None, figsize=(10, 10)):
    
    name = "binned intensity"
    
    resultManager.attempt_load_all()
    
    console.print(header_fmt.format(f"Plotting '{name}' plots..."))

    if bins_to_plot is None:
        bins_to_plot = np.arange(resultManager.n_mass_bins)

    cols_to_plot = ['total'] + resultManager.waveNames
    nrows, ncols = calculate_subplot_grid_size(len(cols_to_plot))
    
    saved_files = []
    for bin in tqdm.tqdm(bins_to_plot):
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        mass_center = resultManager.mass_centers[bin]
        binned_mcmc_samples = resultManager.mcmc_results.query(f'mass == {mass_center}')
        binned_mle_results = resultManager.mle_results.query(f'mass == {mass_center}')
        binned_gen_results = resultManager.gen_results[0].query(f'mass == {mass_center}')
        binned_ift_results = resultManager.ift_results[0].query(f'mass == {mass_center}')
        
        if len(binned_mcmc_samples) == 0:
            raise ValueError(f"No MCMC samples found for bin {bin}")
        if len(binned_mle_results) == 0:
            raise ValueError(f"No MLE results found for bin {bin}")
        if len(binned_gen_results) == 0:
            raise ValueError(f"No generated results found for bin {bin}")
        if len(binned_ift_results) == 0:
            raise ValueError(f"No IFIT results found for bin {bin}")
        
        edges = np.linspace(0, binned_mcmc_samples["total_intensity"].max(), 200)

        for i, wave in enumerate(cols_to_plot):
            irow, icol = i // ncols, i % ncols
            ax = axes[irow, icol]
            
            #### PLOT MCMC RESULTS ####
            ax.hist(binned_mcmc_samples[f"{wave}"], bins=edges, alpha=1.0, color=mcmc_color)
            
            #### PLOT MLE RESULTS ####
            # TODO: after computing errors for intensities, use axvspan
            for irow in binned_mle_results.index: # for each random start MLE fit
                ax.axvline(binned_mle_results.loc[irow, wave], color=mle_color, linestyle='--', alpha=0.1)
            
            #### PLOT GENERATED RESULTS ####
            col = f"{wave}" if wave != "total_intensity" else "fitted_intensity"
            ax.axvline(binned_gen_results[col].values[0], color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=2)
            
            #### PLOT NIFTY FIT RESULTS ####
            mass_center = resultManager.mass_centers[bin]
            _wave = "fitted_intensity" if wave == "total_intensity" else wave
            mean = np.mean(binned_ift_results[_wave].values) # mean over nifty samples
            std = np.std(binned_ift_results[_wave].values)   # std over nifty samples
            ax.axvline(mean, color=ift_color_dict["Signal"], linestyle='--', alpha=1.0, linewidth=1)
            ax.axvspan(mean - std, mean + std, color=ift_color_dict["Signal"], alpha=0.3)

        for ax, wave in zip(axes.flatten(), cols_to_plot):
            ax.set_title(prettyLabels[wave], size=14, color='red', fontweight='bold')
            ax.set_ylim(0)
            ax.set_xlim(0)
        for icol in range(ncols):
            axes[nrows-1, icol].set_xlabel("Intensity", size=12)
        for irow in range(nrows):
            ylabel = "Samples"
            axes[irow, 0].set_ylabel(ylabel, size=12)
            
        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/intensity/bin{bin}_intensities.png"
        save_plot(ofile, fig, axes, overwrite=True, verbose=False)
        saved_files.append(ofile)
        plt.close()
        
    console.print(f"\nSaved '{name}' plots to:")
    for file in saved_files:
        console.print(f"  - {file}")
    console.print(f"\n")

def plot_binned_complex_plane(resultManager: ResultManager, bins_to_plot=None, figsize=(10, 10),
                              max_samples=500):
    
    name = "binned complex plane"
    
    resultManager.attempt_load_all()
    
    console.print(header_fmt.format(f"Plotting '{name}' plots..."))

    if bins_to_plot is None:
        bins_to_plot = np.arange(resultManager.n_mass_bins)

    cols_to_plot = resultManager.waveNames
    nrows, ncols = calculate_subplot_grid_size(len(cols_to_plot))
    
    saved_files = []
    for bin in tqdm.tqdm(bins_to_plot):
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
        
        mass_center = resultManager.mass_centers[bin]
        binned_mcmc_samples = resultManager.mcmc_results.query(f'mass == {mass_center}')
        binned_mle_results = resultManager.mle_results.query(f'mass == {mass_center}')
        binned_gen_results = resultManager.gen_results[0].query(f'mass == {mass_center}')
        binned_ift_results = resultManager.ift_results[0].query(f'mass == {mass_center}')
        
        # share limits so we know some amplitudes are small
        all_amps = np.array([[np.real(binned_mcmc_samples[f'{wave}_amp']), np.imag(binned_mcmc_samples[f'{wave}_amp'])] for wave in cols_to_plot])
        max_lim = max(np.abs(all_amps.max()), np.abs(all_amps.min()))
        bin_edges = np.linspace(-max_lim, max_lim, 100)

        for i, wave in enumerate(cols_to_plot):
            cval = binned_mcmc_samples[f"{wave}_amp"] # [chain_offset*nsamples:(chain_offset+1)*nsamples]
            rval, ival = np.real(cval), np.imag(cval)
            reference_wave = resultManager.sector_to_ref_wave[wave[-1]].strip("_amp")
            is_reference = wave == reference_wave
            reference_intensity = np.array(binned_mcmc_samples[reference_wave])
            norm = plt.Normalize(reference_intensity.min(), reference_intensity.max())
            cmap = plt.cm.inferno
                    
            #### PLOT MCMC RESULTS ####
            if np.all(ival < 1e-5): # if imaginary part is ~ 0 then plot real part as a histogram (i.e. reference waves)
                digitized = np.digitize(rval, bin_edges)
                binned_mcmc_intensities = np.zeros(len(bin_edges)-1)
                for j in range(len(bin_edges)-1):
                    bin_mask = (digitized == j+1)
                    if np.any(bin_mask):
                        binned_mcmc_intensities[j] = np.mean(reference_intensity[bin_mask])
                patches = axes[i].hist(rval, bins=100, alpha=0.7, color='gray')[2]
                for j, patch in enumerate(patches):
                    if j < len(binned_mcmc_intensities):
                        color = cmap(norm(binned_mcmc_intensities[j]))
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                axes[i].set_xlabel("Real", size=12)
                axes[i].set_xlim(-max_lim, max_lim)
                axes[i].set_ylim(0)
            else: # scatter plot complex plane
                rnd = np.random.permutation(np.arange(len(rval)))[:max_samples]
                axes[i].scatter(rval[rnd], ival[rnd], alpha=0.1, c=reference_intensity[rnd], cmap=cmap, norm=norm)
                axes[i].set_xlabel("Real", size=12)
                axes[i].set_ylabel("Imaginary", size=12)
                axes[i].set_xlim(-max_lim, max_lim)
                axes[i].set_ylim(-max_lim, max_lim)
                
            #### PLOT MLE RESULTS ####
            for irow in binned_mle_results.index: # for each random start MLE fit
                error_method = 'tikhonov'
                cval = binned_mle_results.loc[irow, f"{wave}_amp"]
                real_part = np.real(cval)
                imag_part = np.imag(cval)
                real_part_semimajor = binned_mle_results.loc[irow, f'{wave}_{error_method}_re_err']
                imag_part_semiminor = binned_mle_results.loc[irow, f'{wave}_{error_method}_im_err']
                
                if wave == is_reference:
                    axes[i].axvspan(real_part - real_part_semimajor, real_part + real_part_semimajor, color=mle_color, alpha=0.01)
                    axes[i].axvline(real_part, color=mle_color, linestyle='--', alpha=0.5, linewidth=1)
                else:
                    part_angle = binned_mle_results.loc[irow, f'{wave}_{error_method}_err_angle'] * 180 / np.pi
                    ellipse = Ellipse(xy=(real_part, imag_part), width=2 * real_part_semimajor, height=2 * imag_part_semiminor, angle=part_angle, facecolor='none', edgecolor='xkcd:red', alpha=0.5)
                    axes[i].add_patch(ellipse)    
            
            #### PLOT GENERATED RESULTS ####
            gen_amp = binned_gen_results[f"{wave}_amp"].values[0]
            if is_reference:
                axes[i].axvline(np.real(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=2)
            else:
                axes[i].axhline(np.imag(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0)
                axes[i].axvline(np.real(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0)
                    
            #### PLOT NIFTY FIT RESULTS ####
            real_part = np.real(binned_ift_results[f"{wave}_amp"])
            imag_part = np.imag(binned_ift_results[f"{wave}_amp"])
            cov = np.cov(real_part, imag_part)
            cov_re = cov[0, 0]
            cov_im = cov[1, 1]
            rho = cov[0, 1]
            real_part_semimajor = np.sqrt(cov_re)
            imag_part_semiminor = np.sqrt(cov_im)
            if is_reference:
                axes[i].axvspan(np.mean(real_part) - real_part_semimajor, np.mean(real_part) + real_part_semimajor, color=ift_color_dict["Signal"], alpha=0.2)
                axes[i].axvline(np.mean(real_part), color=ift_color_dict["Signal"], linestyle='--', alpha=1.0, linewidth=1)
            else:
                part_angle = 0.5 * np.arctan2(2 * rho, cov_re - cov_im) * 180 / np.pi
                ellipse = Ellipse(xy=(np.mean(real_part), np.mean(imag_part)), width=2 * real_part_semimajor, height=2 * imag_part_semiminor, angle=part_angle, 
                                    facecolor='none', edgecolor=ift_color_dict["Signal"], alpha=1.0, linewidth=2)
                axes[i].add_patch(ellipse)
                axes[i].scatter(real_part, imag_part, color=ift_color_dict["Signal"], alpha=0.4, marker='o', s=2)

            # Remove the set_title and use text to place title in top right corner
            axes[i].text(0.975, 0.975, prettyLabels[wave],
                        size=20, color='black', fontweight='bold',
                        horizontalalignment='right', verticalalignment='top',
                        transform=axes[i].transAxes)

        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/complex_plane/bin{bin}_complex_plane.png"
        save_plot(ofile, fig, axes, overwrite=True, verbose=False)
        saved_files.append(ofile)
        plt.close()

    console.print(f"\nSaved '{name}' plots to:")
    for file in saved_files:
        console.print(f"  - {file}")
    console.print(f"\n")
    
def montage_and_gif_binned_plots(resultManager: ResultManager):
    
    console.print(header_fmt.format(f"Montaging / GIFing all plots..."))
    
    base_directory = resultManager.base_directory
    
    subdirs = ["complex_plane", "intensity"]
    montage_cmd = "montage {0}/{1}/{2}/bin*.png -density 300 -geometry +10+10 {0}/{1}/{2}/montage_output.png"
    gif_cmd = "convert -delay 40 {0}/{1}/{2}/bin*.png -layers optimize -colors 256 -fuzz 2% {0}/{1}/{2}/output.gif"
    for subdir in subdirs:
        console.print(f"Create montage + gif of plots in '{base_directory}/{default_plot_subdir}/{subdir}'")
        os.system(montage_cmd.format(base_directory, default_plot_subdir, subdir))
        os.system(gif_cmd.format(base_directory, default_plot_subdir, subdir))
        
    subdirs = ["intensity_and_phases"]
    for subdir in subdirs:
        montage_cmd = "montage {0}/{1}/{2}/intensity_phase_plot_*.png -density 300 -geometry +10+10 {0}/{1}/{2}/montage_output.png"
        console.print(f"Create montage of plots in '{base_directory}/{default_plot_subdir}/{subdir}'")
        os.system(montage_cmd.format(base_directory, default_plot_subdir, subdir))
        
    console.print(f"\n")
        
def plot_overview_across_bins(resultManager: ResultManager, n_samples_per_bin=300):

    name = "intensity + phases"
    resultManager.attempt_load_all()
    console.print(header_fmt.format(f"Plotting '{name}' plots..."))

    waveNames = resultManager.waveNames
    n_mass_bins = resultManager.n_mass_bins
    masses = resultManager.masses
    mass_centers = resultManager.mass_centers
    nEvents = resultManager.hist_results['nEvents'].values    
    # TODO: This needs to be fixed, MC without background has incorrect errors
    nEvents_err = resultManager.hist_results['nEvents_err'].values**0.5
    
    bin_width = mass_centers[1] - mass_centers[0]
    line_half_width = bin_width / 2
    
    mcmc_results = resultManager.mcmc_results
    samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.sample(n=n_samples_per_bin, replace=False)).reset_index(drop=True)
    
    mle_results = resultManager.mle_results
    ift_results = resultManager.ift_results[0]
    gen_results = resultManager.gen_results[0]
    
    for k, waveName in enumerate(waveNames):
        
        reference_wave = 'Sp0+' if waveName[-1] == "+" else 'Sp0-'
        
        # Create a new figure for each waveName
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(wspace=0, hspace=0.03)
        
        # Main intensity plot
        ax = axes[0]
        ax.set_xlim(1.04, 1.72)
        
        #### PLOT DATA HISTOGRAM
        # ax.step(mass_centers, nEvents[0], where='post', color='black', alpha=0.8)
        data_line = hep.histplot((nEvents, masses), ax=ax, color='black', alpha=0.8)
        error_bars = ax.errorbar(mass_centers, nEvents, yerr=nEvents_err, 
                    color='black', alpha=0.8, fmt='o', markersize=2, capsize=3, label="Data")
        
        #### PLOT GENERATED CURVE
        gen_cols = [col for col in gen_results.columns if waveName in col and "_amp" not in col]
        if len(gen_cols) == 0:
            console.print(f"[bold yellow]No generated results found for {waveName}[/bold yellow]")
        for col in gen_cols:
            if "_cf" in col: fit_type = "Bkgnd"
            elif len(col.split("_")) > 1: fit_type = "Param."
            else: fit_type = "Signal"
            ax.plot(gen_results['mass'], gen_results[col], color='white',  linestyle='-', alpha=0.3, linewidth=4, zorder=9)
            ax.plot(gen_results['mass'], gen_results[col], color=ift_color_dict[fit_type],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)

        #### PLOT MCMC FIT INTENSITY
        for bin_idx in range(n_mass_bins):
            mass = mass_centers[bin_idx]
            binned_samples = samples_to_draw.query('mass == @mass')
            x_starts = np.full_like(binned_samples[waveName], mass - line_half_width)
            x_ends   = np.full_like(binned_samples[waveName], mass + line_half_width)
            ax.hlines(y=binned_samples[waveName],   
                    xmin=x_starts,
                    xmax=x_ends,
                    colors=mcmc_color,
                    alpha=0.1,
                    linewidth=1)
        mcmc_legend_line = ax.plot([], [], color=mcmc_color, alpha=0.7, linewidth=2, label="MCMC")[0]
            
        #### PLOT NIFTY FIT INTENSITY
        ift_legend_lines = []
        ift_cols = [col for col in ift_results.columns if waveName in col and "_amp" not in col]
        if len(ift_cols) == 0:
            console.print(f"[bold yellow]No IFIT results found for {waveName}[/bold yellow]")
        ift_nsamples = ift_results['sample'].unique().size
        for col in ift_cols:
            if "_cf" in col: fit_type = "Bkgnd"
            elif len(col.split("_")) > 1: fit_type = "Param."
            else: fit_type = "Signal"
            for isample in range(ift_nsamples):
                tmp = ift_results.query('sample == @isample')
                ax.plot(tmp['mass'], tmp[col], color=ift_color_dict[fit_type], 
                                  linestyle='-', alpha=ift_alpha, linewidth=1, label=fit_type)
                if isample == 0: # create empty plot just for the legend to have lines with different alpha
                    ift_line = ax.plot([], [], color=ift_color_dict[fit_type], linestyle='-', alpha=1.0, linewidth=1, label=fit_type)[0]
                    ift_legend_lines.append(ift_line)
        
        #### PLOT MLE FIT INTENSITY
        jitter_scale = mle_jitter_scale * bin_width
        mass_jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(mle_results)) # shared with phases below
        ax.errorbar(mle_results['mass'] + mass_jitter, mle_results[waveName], yerr=mle_results[f"{waveName}_error"], 
                    color=mle_color, alpha=0.2, fmt='o', markersize=2, capsize=3)
        mle_bars = ax.plot([], [], label="MLE", color=mle_color, alpha=1.0,  markersize=2)[0]
        
        # Create the legend with the style from iftpwa_plot.py
        handles = [error_bars, mcmc_legend_line, mle_bars] + ift_legend_lines
        ax.legend(handles=handles, labelcolor="linecolor", handlelength=0.3, 
                 handletextpad=0.15, frameon=False, loc='upper right', prop={'size': 16})
        
        #### PLOT MCMC PHASES
        phase_ax = axes[1]
        for bin_idx in range(n_mass_bins):
            mass = mass_centers[bin_idx]
            binned_samples = samples_to_draw.query('mass == @mass')
            x_starts = np.full_like(binned_samples[waveName], mass - line_half_width)
            x_ends   = np.full_like(binned_samples[waveName], mass + line_half_width)
            phases = np.angle(binned_samples[f"{waveName}_amp"], deg=True)
            phase_ax.hlines(y=phases,   
                    xmin=x_starts, 
                    xmax=x_ends, 
                    colors=mcmc_color, 
                    alpha=0.1, 
                    linewidth=1)
        phase_ax.set_ylim(-180, 180)
        phase_ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        #### PLOT GENERATED PHASES
        phase = np.angle(gen_results[f"{waveName}_amp"], deg=True)
        phase = np.unwrap(phase, period=360)
        for offset in [-360, 0, 360]:
            phase_ax.plot(gen_results['mass'], phase + offset, color='white',  linestyle='-', alpha=0.3, linewidth=4, zorder=9)
            phase_ax.plot(gen_results['mass'], phase + offset, color=ift_color_dict["Signal"],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)
        
        #### PLOT NIFTY PHASES
        for isample in range(ift_nsamples):
            tmp = ift_results.query('sample == @isample')
            phase = np.angle(tmp[f"{waveName}_amp"], deg=True)
            phase = np.unwrap(phase, period=360)
            for offset in [-360, 0, 360]:
                phase_ax.plot(tmp['mass'], phase + offset, color=ift_color_dict["Signal"], linestyle='-', alpha=ift_alpha, linewidth=1)
            
        #### PLOT MLE PHASES
        phase = mle_results[f"{waveName}_relative_phase"]
        phase_error = mle_results[f"{waveName}_relative_phase_error"]
        phase_ax.errorbar(mle_results['mass'] + mass_jitter, phase, yerr=phase_error, 
                          color=mle_color, alpha=0.2, fmt='o', markersize=2, capsize=3)
            
        #### PLOT WAVE NAME IN CORNER
        text = ax.text(0.05, 0.87, f"{prettyLabels[waveName]}", transform=ax.transAxes, fontsize=36, c='black', zorder=9)
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        
        #### AXIS SETTINGS
        ax.set_ylim(0)
        phase_ax.set_xlabel(r"$m_X$ [GeV]", size=20)
        phase_ax.tick_params(axis='x', labelsize=16)
        ax.set_ylabel(rf"Intensity / {bin_width:.2f} GeV", size=22)
        ax.tick_params(axis='y', labelsize=16)
        wave_label1 = prettyLabels[waveName].strip("$")
        wave_label2 = prettyLabels[reference_wave].strip("$")
        phase_ax.set_ylabel(f"$\phi_{{{wave_label1}}} - \phi_{{{wave_label2}}}$ [deg]", size=18)

        # Save each figure to its own PNG file
        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/intensity_and_phases/intensity_phase_plot_{waveName}.png"
        save_plot(ofile, fig, axes, overwrite=True, verbose=True)
        plt.close()
        
    console.print(f"\n")