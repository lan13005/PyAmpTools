import os
os.environ["JAX_PLATFORMS"] = "cpu"
from pyamptools.utility.general import load_yaml, calculate_subplot_grid_size, prettyLabels, identify_channel # TODO: must be loaded before loadIFTResultsFromPkl, IDKY yet
from pyamptools.utility.MomentUtilities import MomentManagerTwoPS, MomentManagerVecPS
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
import tqdm
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects
import mplhep as hep
import gc
import io

# TODO:
# - Ensure all IFT, MLE, MCMC uses intensity, intensity_error
# - Move GENERATED from RESULTS_MC - prior sim function needs to be created to do this
# - symlink all plot subdirectories to RESULTS folder? (i.e. nifty results)
# - implement t-dependence?
# - ResultsManager can already do this, just move output helpful comments from resultDump.py into the manager
# - add guards to not plot things that do not exist in the plotting functions

# NOTE:
# - Code allows users to load additional results from different sources manually using load_*_results functions
#   Users must provide an alias (source name) which will be added to the loaded dataframe and then pd.concat will be called

console = Console()

binNum_regex = re.compile(r'bin(\d+)')
header_fmt = "[bold blue]***************************************************\n{}\n[/bold blue]"

mle_color = 'xkcd:orange'
ift_color_dict = {
    "Signal": "xkcd:sea blue",
    "Param": "xkcd:cerise",
    "Bkgnd": "xkcd:dark sea green"
}
gen_color = 'xkcd:green'
mcmc_color = 'black'
moment_inversion_color = 'xkcd:purple'

ift_alpha = 0.45 # transparency of IFT fit results
mle_jitter_scale = 0.1 # percent of the bin width the MLE results are jittered

# All plots are stored in this directory within the `base_directory` key in the YAML file
default_plot_subdir = "PLOTS"

class ResultManager:
    
    def __init__(self, main_yaml, silence=False):
        """
        Args:
            main_yaml [str, Dict]: path to the main yaml file or already loaded as a dictionary
            verbose [bool]: whether to print verbose output to console
        """
        
        # Set up console based on verbose flag
        self.silence = silence
        self.console = console
        if silence: # silence output even for jupyter notebooks
            self.console = Console(file=io.StringIO(), force_terminal=True, force_jupyter=False)
        
        self._mle_results  = pd.DataFrame()
        self._gen_results  = [pd.DataFrame(), pd.DataFrame()] # (generated samples of amplitudes, generated samples of resonance parameters)
        self._ift_results  = [pd.DataFrame(), pd.DataFrame()] # (fitted samples of amplitudes, fitted samples of resonance parameters)
        self._mcmc_results = pd.DataFrame()
        self._hist_results = pd.DataFrame()
        self._moment_inversion_results = pd.DataFrame()
        self._reloaded_normalization_scheme = None # if reloading from cache, this variable will store used normalization scheme
        
        self.main_dict = main_yaml
        if isinstance(main_yaml, str):
            self.main_dict = load_yaml(main_yaml)
        self.iftpwa_dict = self.main_dict['nifty']['yaml']
        self.base_directory = self.main_dict['base_directory']
        self.waveNames = self.main_dict['waveset'].split("_")
        self.phase_reference = self.main_dict['phase_reference'].split("_")
        min_mass = self.main_dict['min_mass']
        max_mass = self.main_dict['max_mass']

        self.n_mass_bins = self.main_dict['n_mass_bins']
        self.massBins = np.linspace(min_mass, max_mass, self.n_mass_bins+1)
        self.masses = (self.massBins[:-1] + self.massBins[1:]) / 2
        self.mass_bin_width = self.massBins[1] - self.massBins[0]
        
        self.n_t_bins = self.main_dict['n_t_bins']
        self.ts = np.linspace(self.main_dict['min_t'], self.main_dict['max_t'], self.n_t_bins+1)
        self.t_centers = (self.ts[:-1] + self.ts[1:]) / 2
        self.t_bin_width = self.ts[1] - self.ts[0]
        
        # Round all floats in kinematic binning
        self.n_decimal_places = 5
        self.massBins = np.round(self.massBins, self.n_decimal_places)
        self.masses = np.round(self.masses, self.n_decimal_places)
        self.ts = np.round(self.ts, self.n_decimal_places)
        self.t_centers = np.round(self.t_centers, self.n_decimal_places)
        self.mass_bin_width = np.round(self.mass_bin_width, self.n_decimal_places)
        self.t_bin_width = np.round(self.t_bin_width, self.n_decimal_places)
        
        self.channel = identify_channel(self.waveNames) # ~str: TwoPseudoscalar or VectorPseudoscalar
        self.moment_latex_dict = None
        
        # Moment calculations can take a long time (especially lots of MCMC samples, we can cache the result)
        #    NOTE: This will technically cache the entire dataframe (including amplitudes, etc)
        self.moment_cache_location = f"{self.base_directory}/projected_moments_cache.pkl"
        
        # TODO: fix bins per group
        self.bpg = self.main_dict['bins_per_group']
        
        n_t_bins = self.main_dict['n_t_bins']
        if n_t_bins != 1:
            self.console.print(f"[bold yellow]warning: Default plotting scripts will not work with more than 1 t-bin. Data loading should be fine[/bold yellow]")
        
        # Keep track of all the unique fit sources that the user loads
        #   Why would the user want to load multiple gen/hist sources?
        self.source_list = {
            'mle': [],
            'mcmc': [],
            'ift_GENERATED': [],
            'ift_FITTED': [],
            'hist': [],
            'gen': [],
            'moment_inversion': [],
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
        
        self.console.print(f"\n")
        self.console.print(header_fmt.format(f"Parsing main_yaml with these expected settings:"))
        self.console.print(f"wave_names: {self.waveNames}")
        self.console.print(f"identified {len(self.sectors)} incoherent sectors: {self.sectors}")
        self.console.print(f"n_mass_bins: {self.n_mass_bins}")
        self.console.print(f"masses: {self.masses}")
        self.console.print(f"\n")
        
    def attempt_load_all(self):
        
        if self._hist_results.empty:
            self._hist_results = self.load_hist_results()
            
        if os.path.exists(self.moment_cache_location):
            self.console.print(f"\nLoading cached data with projected moments (and original amplitudes, etc) from {self.moment_cache_location}\n", style="bold dark_orange")
            with open(self.moment_cache_location, "rb") as f:
                _results = pkl.load(f)
                
                self._reloaded_normalization_scheme = _results["normalization_scheme"]
                
                if "mle"  in _results and not _results["mle"].empty: self._mle_results  = _results["mle"]
                else: self._mle_results  = self.load_mle_results()
                    
                if "mcmc" in _results and not _results["mcmc"].empty: self._mcmc_results = _results["mcmc"]
                else: self._mcmc_results = self.load_mcmc_results()
                    
                if "ift"  in _results and not _results["ift"][0].empty: self._ift_results  = _results["ift"]
                else: self._ift_results  = self.load_ift_results(source_type="FITTED")
                    
                if "gen"  in _results and not _results["gen"][0].empty: self._gen_results  = _results["gen"]
                else: self._gen_results  = self.load_ift_results(source_type="GENERATED")
                
                if "moment_inversion" in _results and not _results["moment_inversion"].empty: self._moment_inversion_results = _results["moment_inversion"]
                else: self._moment_inversion_results = self.load_moment_inversion_results()
                
                if "moment_latex_dict" in _results: self.moment_latex_dict = _results["moment_latex_dict"]
                else: self.moment_latex_dict = None
            return
        else:
            self._mle_results  = self.load_mle_results()
            self._mcmc_results = self.load_mcmc_results()
            self._ift_results = self.load_ift_results(source_type="FITTED")
            self._gen_results = self.load_ift_results(source_type="GENERATED")
            self._moment_inversion_results = self.load_moment_inversion_results()
            
    def attempt_project_moments(self, normalization_scheme=0, pool_size=-1, silence=False):
        
        self.console.print(f"User requested moments to be calculated")
        dfs_to_process = [
            ("mle",  self._mle_results ), 
            ("mcmc", self._mcmc_results),
            ("ift",  self._ift_results[0]),
            ("gen",  self._gen_results[0])
        ]
        
        # Load pool size from yaml file if not usable number
        if pool_size < 1:
            pool_size = self.main_dict['n_processes']
        
        _results = {}
        # Project dataframe of partial wave amplitudes into moment basis
        for df_name, df in dfs_to_process:
            # Check if the dataframe has any columns that start with "H0" which is the zeroth moment that is equal to the intensity
            #   This should exist for every channel?
            #   If so, we skip the entire moment projection process. If there was a missing moment, you would need to restart by deleting the cache file
            if df.empty:
                self.console.print(f"'{df_name}' is empty, skipping moment projection")
            else:
                if any(["H0" in col for col in df.columns]):
                    self.console.print(f"'{df_name}' already has projected moments, skipping moment projection")
                else:
                    self.console.print(f"Begin projecting amplitudes onto moment basis for '{df_name}' with {pool_size} processes")
                    if self.channel == "TwoPseudoscalar":
                        momentManager = MomentManagerTwoPS(df, self.waveNames)
                    elif self.channel == "VectorPseudoscalar":
                        momentManager = MomentManagerVecPS(df, self.waveNames)
                    else:
                        raise ValueError(f"Unknown channel for moment projection: {self.channel}")
                    self.console.print(f"  Dataset: '{df_name}'")
                    
                    # Normalize to the intensity in the bin [scheme=1] (acceptance corrected or not depends on your setting in the YAML file when the dataframe was created)
                    processed_df, moment_latex_dict = momentManager.process_and_return_df(
                        normalization_scheme=normalization_scheme,
                        pool_size=pool_size, 
                        append=True,
                        silence=silence
                    )                
                    if   df_name == "mle":  self._mle_results    = processed_df
                    elif df_name == "mcmc": self._mcmc_results   = processed_df
                    elif df_name == "ift":  self._ift_results[0] = processed_df
                    elif df_name == "gen":  self._gen_results[0] = processed_df
                    
                    # Save updated dataframes (with moments) to cache
                    if df_name == 'ift':
                        _results[df_name] = [processed_df, self._ift_results[1]]
                    elif df_name == 'gen':
                        _results[df_name] = [processed_df, self._gen_results[1]]
                    else:
                        _results[df_name] = processed_df
                        
                    # Track dictionary of latex names for plotting moments, should be OK to overwrite
                    self.moment_latex_dict = moment_latex_dict
        
        # The above loop projects moments if they do not exist in the dataframe yet
        #   We update the cache with the previous results (that already had projected moments) with new dataframes with projected moments
        if "mle" not in _results:  _results["mle"]  = self._mle_results
        if "mcmc" not in _results: _results["mcmc"] = self._mcmc_results
        if "ift" not in _results:  _results["ift"]  = self._ift_results
        if "gen" not in _results:  _results["gen"]  = self._gen_results
        if "moment_inversion" not in _results:  _results["moment_inversion"]  = self._moment_inversion_results
        _results["moment_latex_dict"] = self.moment_latex_dict
        _results["normalization_scheme"] = normalization_scheme
        with open(self.moment_cache_location, "wb") as f:
            pkl.dump(_results, f)

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
    
    @property
    def moment_inversion_results(self) -> pd.DataFrame:
        return self._moment_inversion_results
    
    def summarize(self):
        self.console.print(f"Summary of loaded results:")
        self.console.print(f"  resultManager.mle_results: {self.mle_results.shape}")
        self.console.print(f"  resultManager.mcmc_results: {self.mcmc_results.shape}")
        self.console.print(f"  resultManager.ift_results: {self.ift_results[0].shape}")
        self.console.print(f"  resultManager.gen_results: {self.gen_results[0].shape}")
        self.console.print(f"  resultManager.moment_inversion_results: {self.moment_inversion_results.shape}")
    
    def load_hist_results(self, base_directory=None, alias=None):
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
            
        source_name = self._check_and_get_source_name(self._hist_results, 'hist', base_directory, alias)
        if source_name is None:
            return self._hist_results
        
        result_dir = f"{base_directory}/BINNED_DATA"
        if not os.path.exists(result_dir):
            self.console.print(f"[bold red]No 'histogram' results found in {result_dir}, return existing results with shape {self._hist_results.shape}\n[/bold red]")
            return self._hist_results
        
        self.console.print(header_fmt.format(f"Loading 'HISTOGRAM' result from {result_dir}"))
        nEvents, nEvents_err = get_nEventsInBin(result_dir)
        hist_df = {'mass': self.masses, 'nEvents': nEvents, 'nEvents_err': nEvents_err}
        hist_df = pd.DataFrame(hist_df)
        
        self.console.print(f"[bold green]Total Events Histogram Summary:[/bold green]")
        self.console.print(f"columns: {hist_df.columns}")
        self.console.print(f"shape: {hist_df.shape}")
        self.console.print(f"\n")
        
        hist_df = self._add_source_to_dataframe(hist_df, 'hist', source_name)
        if not self._hist_results.empty:
            self.console.print(f"[bold green]Previous histogram results were loaded already, concatenating[/bold green]")
            hist_df = pd.concat([self._hist_results, hist_df])
        return hist_df
    
    def load_ift_results(self, base_directory=None, source_type="GENERATED", alias=None):
        
        """
        Currently only supports generated curves from iftpwa
        
        NOTE: Whenever you want to determine some statistics from IFT you should do it on a sample by sample basis (curve across kinematics bins)
                For instance to calculate integrated intensity (across mass) option 1 is more accurate:
                1. Grouping by sample, sum over mass, then calculate mean / std produces very different results compared to
                2. Grouping by mass and summing over samples. This is due to the fact that the samples are entire curves!
                In one test case, the relative uncertainties went from 20% to 4% where first number comes from grouping by sample
        """
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
            
        if source_type not in ["GENERATED", "FITTED"]:
            raise ValueError(f"Source type {source_type} not supported. Must be one of: [GENERATED, FITTED]")
        
        subdir = "GENERATED" if source_type == "GENERATED" else "NIFTY"
        result_dir = f"{base_directory}/{subdir}"
        self.console.print(header_fmt.format(f"Loading '{source_type}' results from {result_dir}"))
        
        if source_type == "GENERATED":
            ift_results = self._gen_results
        else:
            ift_results = self._ift_results
        
        source_name = self._check_and_get_source_name(ift_results, f'ift_{source_type}', base_directory, alias)
        if source_name is None:
            return ift_results
        
        truth_loc = f"{result_dir}/niftypwa_fit.pkl"
        if not os.path.exists(truth_loc):
            self.console.print(f"[bold red]No '{subdir.lower()}' curves found in {truth_loc}, return existing results with shape {ift_results[0].shape}\n[/bold red]")
            return ift_results
        with open(truth_loc, "rb") as f:
            truth_pkl = pkl.load(f)
        ift_df, ift_res_df = loadIFTResultsFromPkl(truth_pkl)

        # requires loading hist results first to scale generated curves to data scale
        if source_type == "GENERATED":
            self._hist_results = self.load_hist_results()
            if self._hist_results is None:
                raise ValueError("Histogram results not found. We use this dataset to scale the generated curves, cannot proceed without it.")
            mle_totals = self._hist_results['nEvents'].sum()
            rescaling = mle_totals / np.sum(ift_df['intensity'])
            amp_cols = [c for c in ift_df.columns if c.endswith('_amp')]
            intensity_cols = [c for c in ift_df.columns if 'intensity' in c] + [amp.strip('_amp') for amp in amp_cols]
            ift_df[amp_cols] *= np.sqrt(rescaling)
            ift_df[intensity_cols] *= rescaling
        
        if len(ift_df) == 0:
            self.console.print(f"[bold red]No '{subdir.lower()}' curves found in {truth_loc}, return existing results with shape {ift_results[0].shape}\n[/bold red]")
            return ift_results
        
        # rotate away reference waves
        ift_df = self._rotate_away_reference_waves(ift_df)
        ift_df['mass'] = np.round(ift_df['mass'], self.n_decimal_places)
        
        ift_df = self._add_source_to_dataframe(ift_df, f'ift_{source_type}', source_name)
        ift_res_df = self._add_source_to_dataframe(ift_res_df, None, source_name) # already appeneded above, dont track again
        if not ift_results[0].empty:
            self.console.print(f"[bold green]Previous IFT results were loaded already, concatenating[/bold green]")
            ift_df = pd.concat([ift_results[0], ift_df])        
            ift_res_df = pd.concat([ift_results[1], ift_res_df])
            
        # NOTE: These print statements should come before self.hist_results is called since additional results will be loaded and printed
        self.console.print(f"[bold green]\nIFT {subdir} Summary:[/bold green]")
        self.console.print(f"Amplitudes DataFrame columns: {list(ift_df.columns)}")
        self.console.print(f"Amplitudes DataFrame shape: {ift_df.shape}")
        self.console.print(f"Resonance Parameters DataFrame columns: {list(ift_res_df.columns)}")
        self.console.print(f"Resonance Parameters DataFrame shape: {ift_res_df.shape}")
        self.console.print(f"\n")
    
        return [ift_df, ift_res_df]
        
    def load_mle_results(self, base_directory=None, alias=None):
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
        
        result_dir = f"{base_directory}/MLE"
        self.console.print(header_fmt.format(f"Loading 'MLE' results from {result_dir}"))
        
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

        def load_from_pkl_list(pkl_list, masses, waveNames):
            
            pkl_list = sorted(pkl_list, key=lambda x: int(binNum_regex.search(x).group(1)))

            results = {}
                        
            for pkl_file in pkl_list:
                with open(pkl_file, "rb") as f:
                    datas = pkl.load(f)
                    
                self.console.print(f"Loaded {len(datas)} fits with random starts from {pkl_file}")
                    
                bin_idx = int(pkl_file.split("_")[-1].lstrip("bin").rstrip(".pkl"))

                for i, data in enumerate(datas):

                    results.setdefault("mass", []).append(masses[bin_idx])
                    results.setdefault("initial_likelihood", []).append(data["initial_likelihood"])
                    results.setdefault("likelihood", []).append(data["likelihood"])
                    results.setdefault("sample", []).append(i)
                    
                    for key in data["final_par_values"].keys():
                        results.setdefault(key, []).append(data["final_par_values"][key])
                    
                    results.setdefault("intensity", []).append(data["intensity"])
                    results.setdefault("intensity_err", []).append(data["intensity_err"])
                    for waveName in waveNames:
                        results.setdefault(f"{waveName}", []).append(data[f"{waveName}"])
                        results.setdefault(f"{waveName}_err", []).append(data[f"{waveName}_err"])
                    
                     # Use this regularized covariance so we always have some error estimate
                     # Obtained by shifting Hessian by smallest eigenvalue (to make positive definite)
                     #    Then inverting to get covariance
                    method = "tikhonov"
                    for iw, wave in enumerate(waveNames):
                        tag = f"{wave}"
                        covariance = data['covariances'][method]
                        reference_wave = self.sector_to_ref_wave[wave[-1]]
                        if covariance is None: # this can happen for Minuit if it fails to converge (IDK about scipy)
                            results.setdefault(f'{tag}_re_err', []).append(None)
                            results.setdefault(f'{tag}_im_err', []).append(None)
                            results.setdefault(f'{tag}_err_angle', []).append(None)
                        else:
                            assert covariance.shape == (2 * len(waveNames), 2 * len(waveNames))
                            
                            # NOTE: The following is not really necessary if our reference wave only allows for [0, pi] values, angles are preserved
                            #       Leave this code here in case things change?
                            # 1. Extract 2x2 submatrix for real/imag parts of ONE amplitude
                            # 2. Create orthogonal SO(2) rotation matrix to rotate 2x2 real/imag part submatrix
                            #    There is no error propagation from the uncertainty in the reference wave. Too complicated. We have MCMC/IFT samples anyways
                            # 3. Perform eigendecomposition, get major/minor axes and angle (of the major axis)
                            indices = [2*iw, 2*iw+1]
                            submatrix = covariance[np.ix_(indices, indices)]
                            
                            reference_wave = self.sector_to_ref_wave[wave[-1]] # contains '_amp' suffix
                            jw = waveNames.index(reference_wave.strip("_amp")) # waveNames has no suffix                            
                            amp2 = data['final_par_values'][f"{reference_wave}"]
                            ref_phase = np.angle(amp2)
                            cos_phase = np.cos(ref_phase)
                            sin_phase = np.sin(ref_phase)
                            rot = np.array([
                                [cos_phase, sin_phase],
                                [-sin_phase, cos_phase]
                            ])                            
                            rotated_submatrix = rot @ submatrix @ rot.T # apply rotation on submatrix
                            
                            # Perform eigendecomposition on the rotated covariance matrix
                            eigenvalues, eigenvectors = np.linalg.eigh(rotated_submatrix) # eigenvalues in ascending order, eigenvectors as columns [:, i]
                            eigenvalues = np.maximum(0, eigenvalues) # Covariance matrices are positive semi-definite so IDK if I need this
                            major_axis = np.sqrt(eigenvalues[1])  # Largest eigenvalue
                            minor_axis = np.sqrt(eigenvalues[0])  # Smallest eigenvalue                            
                            angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]) # atan2 of opposite / adjacent
                            
                            # I think this is correct? We can still call the real/imag axes as the major/minor axes since we also compute an ellipse angle
                            results.setdefault(f'{tag}_re_err', []).append(major_axis)
                            results.setdefault(f'{tag}_im_err', []).append(minor_axis)
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
                            results.setdefault(f'{wave}_relative_phase_err', []).append(relative_phase_error)
                    
            results = pd.DataFrame(results)
            
            if len(results) == 0:
                self.console.print(f"[bold red]No results found in pkl files for: {pkl_list}\n[/bold red]")
                return pd.DataFrame()
            
            results = results.sort_values(by=["mass"]).reset_index(drop=True)
            
            # rotate away reference waves
            results = self._rotate_away_reference_waves(results)
            return results
        
        pkl_list = glob.glob(f"{result_dir}/*pkl")
        if len(pkl_list) == 0:
            self.console.print(f"[bold red]No 'MLE' results found in {result_dir}, return existing results with shape {self._mle_results.shape}\n[/bold red]")
            return self._mle_results
        
        mle_results = load_from_pkl_list(pkl_list, self.masses, self.waveNames)
        
        if mle_results.empty:
            self.console.print(f"[bold red]No 'MLE' results found in {result_dir}, return existing results with shape {self._mle_results.shape}\n[/bold red]")
            return self._mle_results
        
        self.console.print(f"[bold green]\nMLE Summary:[/bold green]")
        self.console.print(f"columns: {list(mle_results.columns)}")
        self.console.print(f"n_random_starts: {mle_results.shape[0] // self.n_mass_bins}")
        self.console.print(f"shape: {mle_results.shape} ~ (n_bins * n_random_starts, columns)")        
        self.console.print(f"\n")
        
        mle_results['mass'] = np.round(mle_results['mass'], self.n_decimal_places)

        mle_results = self._add_source_to_dataframe(mle_results, 'mle', source_name)
        if not self._mle_results.empty:
            self.console.print(f"[bold green]Previous MLE results were loaded already, concatenating[/bold green]")
            mle_results = pd.concat([self._mle_results, mle_results])        
        return mle_results
        
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
        
        result_dir = f"{base_directory}/MCMC"
        self.console.print(header_fmt.format(f"Loading 'MCMC' results from {result_dir}"))
        
        source_name = self._check_and_get_source_name(self._mcmc_results, 'mcmc', base_directory, alias)
        if source_name is None:
            return self._mcmc_results
        
        pkl_list = glob.glob(f"{result_dir}/*_samples.pkl")
        if len(pkl_list) == 0:
            self.console.print(f"[bold red]No 'MCMC' results found in {result_dir}, return existing results with shape {self._mcmc_results.shape}\n[/bold red]")
            return self._mcmc_results

        with open(pkl_list[0], "rb") as f:
            mcmc_results = pkl.load(f)
        mcmc_results = pd.DataFrame(mcmc_results)
        
        if len(mcmc_results) == 0:
            self.console.print(f"[bold red]No 'MCMC' results found in {result_dir}, return existing results with shape {self._mcmc_results.shape}\n[/bold red]")
            return self._mcmc_results
        
        # rotate away reference waves
        mcmc_results = self._rotate_away_reference_waves(mcmc_results)
        mcmc_results['mass'] = np.round(mcmc_results['mass'], self.n_decimal_places)
        
        mcmc_results = self._add_source_to_dataframe(mcmc_results, 'mcmc', source_name)
        if not self._mcmc_results.empty:
            self.console.print(f"[bold green]Previous MCMC results were loaded already, concatenating[/bold green]")
            mcmc_results = pd.concat([self._mcmc_results, mcmc_results])
        
        # print diagnostics
        self.console.print(f"[bold green]\nMCMC Summary:[/bold green]")
        self.console.print(f"columns: {list(mcmc_results.columns)}")
        self.console.print(f"n_samples: {mcmc_results.shape[0] // self.n_mass_bins}")
        self.console.print(f"shape: {mcmc_results.shape} ~ (n_samples * n_bins, columns)")
        self.console.print(f"\n")

        return mcmc_results
    
    def load_moment_inversion_results(self, base_directory=None, alias=None):
        """
        Loads moment inversion results from a given base_directory
        
        Args:
            base_directory: str, path to the base directory containing 'MOMENT_INVERSION' subdirectory.
                If None, will load from self.base_directory specified in the yaml file.
            alias: str, optional alias to identify these results. If None and base_directory
                is the default, will use 'yaml' as the source name.
                
        Returns:
            pd.DataFrame, moment inversion results
        """
        
        if base_directory is None:
            base_directory = self.base_directory
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Base directory {base_directory} does not exist!")
            
        result_dir = f"{base_directory}/MOMENT_INVERSION"
        self.console.print(header_fmt.format(f"Loading 'MOMENT_INVERSION' results from {result_dir}"))
        
        source_name = self._check_and_get_source_name(self._moment_inversion_results, 'moment_inversion', base_directory, alias)
        if source_name is None:
            return self._moment_inversion_results
            
        pkl_list = glob.glob(f"{result_dir}/*pkl")
        if len(pkl_list) == 0:
            self.console.print(f"[bold yellow]No 'MOMENT_INVERSION' results found in {result_dir}, return existing results with shape {self._moment_inversion_results.shape}\n[/bold yellow]")
            return self._moment_inversion_results
            
        # Load the first pickle file found
        moment_inversion_results = pd.DataFrame()
        for inversion_file in pkl_list:
            with open(inversion_file, 'rb') as f:
                results = pkl.load(f)
            _moment_inversion_results = results['prediction']
            moment_inversion_results = pd.concat([moment_inversion_results, _moment_inversion_results])
        moment_inversion_results = moment_inversion_results.sort_values(by='mass')
        moment_inversion_results = moment_inversion_results.reset_index(drop=True)
        
        if len(moment_inversion_results) == 0:
            self.console.print(f"[bold yellow]No 'MOMENT_INVERSION' results found in {result_dir}, return existing results with shape {self._moment_inversion_results.shape}\n[/bold yellow]")
            return self._moment_inversion_results
            
        # rotate away reference waves
        moment_inversion_results = self._rotate_away_reference_waves(moment_inversion_results)
        moment_inversion_results['mass'] = np.round(moment_inversion_results['mass'], self.n_decimal_places)
        
        moment_inversion_results = self._add_source_to_dataframe(moment_inversion_results, 'moment_inversion', source_name)
        if not self._moment_inversion_results.empty:
            self.console.print(f"[bold green]Previous MOMENT_INVERSION results were loaded already, concatenating[/bold green]")
            moment_inversion_results = pd.concat([self._moment_inversion_results, moment_inversion_results])
        
        # print diagnostics
        self.console.print(f"[bold green]\nMOMENT_INVERSION Summary:[/bold green]")
        self.console.print(f"columns: {list(moment_inversion_results.columns)}")
        self.console.print(f"n_samples: {moment_inversion_results.shape[0] // self.n_mass_bins}")
        self.console.print(f"shape: {moment_inversion_results.shape} ~ (n_samples * n_bins, columns)")
        self.console.print(f"\n")
        
        return moment_inversion_results
    
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
                    self.console.print(f"[bold green]Default {source_type} results already loaded. Will return existing DataFrame[/bold green]")
                    source_name = None
                    return source_name
            else:
                raise ValueError(f"[bold red]Alias must be provided for the user-specified base_directory: {base_directory}\n[/bold red]")
        else:
            if alias in self.source_list[source_type]:
                raise ValueError(f"[bold red]{source_type} result with alias {alias} has already been loaded. Please choose a unique alias.[/bold red]")
            source_name = alias
        return source_name

    def _add_source_to_dataframe(self, df, source_type, source_name):
        if df is not None and not df.empty and source_name is not None:
            df['source'] = [source_name] * len(df)
        if source_type is not None and source_name is not None: # source_type=None is reserved to indicates to not track the source name (i.e. if already added before ift_res_df)
            self.source_list[source_type].append(source_name)
        return df
    
    def __repr__(self):
        """Return a string representation of the ResultManager object."""
        repr_str = "\n********************************************************************\n"
        repr_str += f" Number of t-bins (centers): {len(self.t_centers)}: {np.array(self.t_centers)}\n"
        repr_str += f" Number of mass-bins (centers): {len(self.masses)}: {np.array(self.masses)}\n"
        repr_str += f" Number of wave names: {len(self.waveNames)}: {self.waveNames}\n"
        repr_str += f" Access below dataframes as attributes [mle_results, mcmc_results, gen_results, hist_results, ift_results]\n"
        repr_str += "********************************************************************\n"
        
        if not self._gen_results[0].empty:
            repr_str += f" Shape of GENERATED DataFrame {self._gen_results[0].shape} with columns: {self._gen_results[0].columns}\n"
        if not self._hist_results.empty:
            repr_str += f" Shape of HISTOGRAM DataFrame {self._hist_results.shape} with columns: {self._hist_results.columns}\n"
        if not self._ift_results[0].empty:
            repr_str += f" Shape of IFT results DataFrame {self._ift_results[0].shape} with columns: {self._ift_results[0].columns}\n"
        if not self._ift_results[1].empty:
            repr_str += f" Shape of IFT resonance DataFrame {self._ift_results[1].shape} with columns: {self._ift_results[1].columns}\n"
        if not self._mle_results.empty:
            repr_str += f" Shape of MLE DataFrame {self._mle_results.shape} with columns: {self._mle_results.columns}\n"
        if not self._mcmc_results.empty:
            repr_str += f" Shape of MCMC DataFrame {self._mcmc_results.shape} with columns: {self._mcmc_results.columns}\n"
        repr_str += "\n********************************************************************"
        return repr_str
    
    def print_schema(self):
        
        if self.silence:
            print("User requested silence. Skipping print_schema.")
            return
        
        print_schema = "\n\n\n## [bold]SUMMARY OF COLLECTED RESULTS[/bold] ##\n"
        print_schema += (
            "\n********************************************************************\n"
            "[bold]SCHEMA:[/bold]\n"
            "- [cyan]t-bins / mass-bins:[/cyan] Bin centers.\n"
        )
        print_schema += (
            "- [cyan]All DataFrames:[/cyan] Quantities per (t, mass, sample).\n"
            "    - [bold]'intensity'[/bold]: Fitted intensity values (whether acceptance corrected or not depends on YAML file at creation of DataFrame). Aggregate over samples to get mean/std.\n"
            "    - [bold]'<wave>_amp'[/bold]: Contain complex amplitude values for the coherent sum of parameteric and correlated field components.\n"
            "    - [bold]'<wave>'[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for the coherent sum of parameteric and correlated field components.\n"
            "    - [bold]'Hi_LM'[/bold]: Moment with index i for given LM quantum number. i.e. H0_00 is the $H_{0}(0,0)$ moment. Different for VectorPseudoscalar\n"
            ""
        )
        
        print_schema += (
            "- [cyan]IFT DataFrames:[/cyan] Quantities per (t, mass, sample).\n"
            "    - [bold]'<wave>_<res>_amp'[/bold]: Contain complex amplitude values for parameteric components.\n"
            "    - [bold]'<wave>_cf_amp'[/bold]: Contain complex amplitude values for (c)orrelated (f)ield components.\n"
            "    - [bold]'<wave>_<res>'[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for parameteric components.\n"
            "    - [bold]'<wave>_cf'[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for (c)orrelated (f)ield components.\n"
            "    - [bold]'<prior_parameter_name>' (Resonance DataFrame) columns[/bold]: Contain the value of the resonance parameter.\n"
        )    
    
        print_schema += (
            "- [cyan]MLE DataFrame:[/cyan] Quantities per (t, mass, random initialization).\n"
            "    - [bold]'nll'[/bold]: likelihoode\n"
            "    - [bold]'status'[/bold]: minuit return status (0=success)\n"
            "    - [bold]'ematrix'[/bold]: error matrix status (3=success)\n"
            "    - [bold]'<wave> err'[/bold]: Error on fitted wave intensity values\n"
            "    - [bold]'<wave>_re_err'[/bold]: Error on real part of complex amplitude\n"
            "    - [bold]'<wave>_im_err'[/bold]: Error on imaginary part of complex amplitude\n"
            "    - [bold]'<wave>_err_angle'[/bold]: Error ellipse angle from covariance of Real/Imag parts\n"
            "    - [bold]'<wave>_relative_phase'[/bold]: Relative phase 'wave' and its reference wave\n"
            "    - [bold]'<wave>_relative_phase_err'[/bold]: Error on relative phase between 'wave' and its reference wave (error not propagated from rotation)\n"
        )
        
        self.console.print(print_schema)
        # Print the representation of the object
        self.console.print(str(self))
    
    def __del__(self):
        """Ensure proper cleanup when the object is deleted"""
        try:
            plt.close('all')
            # Delete references to original dataframes allowing them to be garbage collected
            self._mle_results = None 
            self._mcmc_results = None
            self._hist_results = None
            self._gen_results = None
            self._ift_results = None            
            gc.collect()
        except Exception:
            # Silently continue on failure, good idea?
            pass

########################################################
# PLOTTING FUNCTIONS HERE
########################################################

def save_and_close_fig(output_file_location, fig, axes, console=None, overwrite=False, verbose=True):
    try:
        # Perform checks
        directory = os.path.dirname(output_file_location)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not overwrite and os.path.exists(output_file_location):
            if verbose: 
                output_string = f"[bold red]File {output_file_location} already exists. Set overwrite=True to overwrite.\n[/bold red]"  
                if console is None: print(output_string)
                else: console.print(output_string)
            return
        # Detect unused axes and set them to not visible
        if not isinstance(axes, np.ndarray): # if it was just a single axis
            axes = np.array([axes])
        for ax in axes.flatten():
            if not ax.has_data():
                ax.set_visible(False)
        # Save plot
        fig.savefig(output_file_location)
        if verbose: 
            output_string = f"[bold green]Creating plot at:[/bold green] {output_file_location}"
            if console is None: print(output_string)
            else: console.print(output_string)
    finally: # always executed even if return is called in try block
        plt.close(fig)
    
def query_default(df):
    """ 'yaml' key is reserved for default source """
    return df.query("source == 'yaml'") if not df.empty else pd.DataFrame()
def safe_query(df, query):
    return df.query(query) if not df.empty else pd.DataFrame()
    
def plot_gen_curves(resultManager: ResultManager, figsize=(10, 10), file_type='pdf'):
    ift_gen_df = resultManager.gen_results[0]
    ift_gen_df = query_default(ift_gen_df)
    cols = ift_gen_df.columns
    
    # Guard against missing data
    if ift_gen_df is None or ift_gen_df.empty:
        resultManager.console.print(f"[bold yellow]No 'generated' curves data available. Skipping gen_curves plot.[/bold yellow]")
        return
    
    resultManager.console.print(f"\n[bold blue]Plotting 'generated curves' plots...[/bold blue]")
    
    waveNames = resultManager.waveNames
    masses = resultManager.masses
        
    nrows, ncols = calculate_subplot_grid_size(len(waveNames))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    for i, wave in enumerate(waveNames):
        irow = i // ncols
        icol = i % ncols
        wave_fit_fraction = ift_gen_df[wave] / ift_gen_df['intensity']
        axes[irow, icol].plot(masses, wave_fit_fraction, color=ift_color_dict['Signal'], label="Signal")
        for col in cols:
            col_parts = col.split("_")
            if wave in col and len(col_parts) > 1 and "amp" not in col_parts:
                wave_fit_fraction = ift_gen_df[col] / ift_gen_df['intensity']
                color = ift_color_dict['Bkgnd'] if "cf" in col_parts else ift_color_dict['Param']
                label = 'Bkgnd' if "cf" in col_parts else 'Param.'
                axes[irow, icol].plot(masses, wave_fit_fraction, color=color, label=label)
        axes[irow, icol].set_ylim(0, 1.1) # fit fractions, include a bit of buffer
        axes[irow, icol].set_xlim(masses[0], masses[-1])
        axes[irow, icol].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[irow, icol].tick_params(axis='both', which='major', labelsize=13)
        axes[irow, icol].text(0.2, 0.975, prettyLabels[wave],
                    size=20, color='black', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top',
                    transform=axes[irow, icol].transAxes)
    axes[0,0].legend(loc='upper right', fontsize=13)
        
    for i in range(ncols):
        axes[nrows-1, i].set_xlabel(r"$m_X$", size=15)
    for i in range(nrows):
        axes[i, 0].set_ylabel("Fit Fraction", size=15)

    plt.tight_layout()
    ofile = f"{resultManager.base_directory}/{default_plot_subdir}/gen_curves.{file_type}"
    save_and_close_fig(ofile, fig, axes, console=resultManager.console, overwrite=True)
    
def plot_binned_intensities(resultManager: ResultManager, bins_to_plot=None, figsize=(10, 10), file_type='pdf', silence=False):
    name = "binned intensity"
    resultManager.console.print(header_fmt.format(f"Plotting '{name}' plots..."))
    
    default_mcmc_results = query_default(resultManager.mcmc_results)
    default_mle_results  = query_default(resultManager.mle_results)
    default_gen_results  = query_default(resultManager.gen_results[0])
    default_ift_results  = query_default(resultManager.ift_results[0])
    default_moment_inversion_results = query_default(resultManager.moment_inversion_results)
    
    # If everything is missing, nothing to do
    if default_mcmc_results.empty and default_mle_results.empty and default_gen_results.empty and default_ift_results.empty and default_moment_inversion_results.empty:
        resultManager.console.print(f"[bold yellow]No data available for binned intensity plots. Skipping.[/bold yellow]")
        return
    
    if bins_to_plot is None:
        bins_to_plot = np.arange(resultManager.n_mass_bins)

    cols_to_plot = ['intensity'] + resultManager.waveNames
    nrows, ncols = calculate_subplot_grid_size(len(cols_to_plot))
    
    saved_files = []
    for bin in tqdm.tqdm(bins_to_plot, disable=silence):
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        mass = resultManager.masses[bin]
        binned_mcmc_samples = safe_query(default_mcmc_results, f'mass == {mass}')
        binned_mle_results  = safe_query(default_mle_results,  f'mass == {mass}')
        binned_gen_results  = safe_query(default_gen_results,  f'mass == {mass}')
        binned_ift_results  = safe_query(default_ift_results,  f'mass == {mass}')
        binned_moment_inversion_results = safe_query(default_moment_inversion_results, f'mass == {mass}')

        # Skip bin if no data available for this mass bin
        if (binned_mcmc_samples.empty and binned_mle_results.empty and 
            binned_gen_results.empty and binned_ift_results.empty and binned_moment_inversion_results.empty):
            resultManager.console.print(f"[bold yellow]No data available for bin {bin} (mass {mass}). Skipping this bin.[/bold yellow]")
            plt.close()
            continue
        
        # bins = np.linspace(0, binned_mcmc_samples["intensity"].max(), 200)
        bins = 200

        for i, wave in enumerate(cols_to_plot):
            irow, icol = i // ncols, i % ncols
            ax = axes[irow, icol]
            
            handles = []
            
            #### PLOT MCMC RESULTS ####
            if not binned_mcmc_samples.empty and wave in binned_mcmc_samples.columns:
                ax.hist(binned_mcmc_samples[f"{wave}"], bins=bins, alpha=1.0, color=mcmc_color)
                mcmc_legend_line = ax.plot([], [], color=mcmc_color, alpha=0.7, linewidth=2, label="mcmc")[0]
                handles.append(mcmc_legend_line)
            
            #### PLOT MLE RESULTS ####
            # TODO: after computing errors for intensities, use axvspan
            if not binned_mle_results.empty and wave in binned_mle_results.columns:
                for irow in range(len(binned_mle_results)): # for each random start MLE fit
                    mean = binned_mle_results.iloc[irow][wave]
                    error = binned_mle_results.iloc[irow][f"{wave}_err"]
                    ax.axvline(mean, color=mle_color, linestyle='--', alpha=0.5)
                    ax.axvspan(mean - error, mean + error, color=mle_color, alpha=0.01)
                mle_bars = ax.plot([], [], label="mle", color=mle_color, alpha=1.0, markersize=2)[0]
                handles.append(mle_bars)
            
            #### PLOT GENERATED RESULTS ####
            col = f"{wave}"
            if not binned_gen_results.empty and col in binned_gen_results.columns:
                ax.axvline(binned_gen_results[col].values[0], color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=2)
                gen_line = ax.plot([], [], color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=1, label="gen")[0]
                handles.append(gen_line)
            
            #### PLOT NIFTY FIT RESULTS ####
            mass = resultManager.masses[bin]
            if not binned_ift_results.empty and wave in binned_ift_results.columns:
                mean = np.mean(binned_ift_results[wave].values) # mean over nifty samples
                std  = np.std(binned_ift_results[wave].values)  # std over nifty samples
                ax.axvline(mean, color=ift_color_dict["Signal"], linestyle='--', alpha=1.0, linewidth=1)
                ax.axvspan(mean - std, mean + std, color=ift_color_dict["Signal"], alpha=0.3)
                ift_line = ax.plot([], [], color=ift_color_dict["Signal"], linestyle='--', alpha=1.0, linewidth=1, label="ift")[0]
                handles.append(ift_line)
            
            #### PLOT MOMENT INVERSION RESULTS ####
            if not binned_moment_inversion_results.empty and wave in binned_moment_inversion_results.columns:
                for irow in range(len(binned_moment_inversion_results)):
                    ax.axvline(binned_moment_inversion_results.iloc[irow][wave], color=moment_inversion_color, linestyle='-', alpha=0.1, linewidth=0.1)
                moment_inversion_line = ax.plot([], [], color=moment_inversion_color, alpha=1.0, linewidth=1, label="mom inv")[0]
                handles.append(moment_inversion_line)

        for ax, wave in zip(axes.flatten(), cols_to_plot):
            ax.set_title(prettyLabels[wave], size=14, color='red', fontweight='bold')
            ax.set_ylim(0)
        for icol in range(ncols):
            axes[nrows-1, icol].set_xlabel("Intensity", size=12)
        for irow in range(nrows):
            ylabel = "Samples"
            axes[irow, 0].set_ylabel(ylabel, size=12)
            
        axes[0, 0].legend(handles=handles, loc='upper right', prop={'size': 8})
            
        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/intensity/bin{bin}_intensities.{file_type}"
        save_and_close_fig(ofile, fig, axes, console=resultManager.console, overwrite=True, verbose=False)
        saved_files.append(ofile)
        plt.close()
        
    resultManager.console.print(f"\nSaved '{name}' plots to:")
    for file in saved_files:
        resultManager.console.print(f"  - {file}")
    resultManager.console.print(f"\n")

def plot_binned_complex_plane(resultManager: ResultManager, bins_to_plot=None, figsize=(10, 10), 
                              mcmc_nsamples=500, mcmc_selection="thin", file_type='pdf', silence=False):
    
    """
    Plot ~posterior distribution of the complex plane, overlaying generated curves, MLE, MCMC, IFT results when possible
    
    Args:
        resultManager: ResultManager instance
        bins_to_plot: List of bin indices to plot. If None, all bins are plotted
        figsize: Tuple of the figure size
        mcmc_nsamples: Number of MCMC samples to scatter (can get too dense if too many are plotted)
        mcmc_selection: ["random", "thin", "last"] mcmc_nsamples will be selected by this criteria
            "random": random samples will be used
            "thin": draw mcmc_nsamples using nsamples / mcmc_nsamples steps
            "last": last mcmc_nsamples will be used
    """
    
    name = "binned complex plane"
    
    resultManager.console.print(header_fmt.format(f"Plotting '{name}' plots..."))
    
    default_mcmc_results = query_default(resultManager.mcmc_results)
    default_mle_results  = query_default(resultManager.mle_results)
    default_gen_results  = query_default(resultManager.gen_results[0])
    default_ift_results  = query_default(resultManager.ift_results[0])
    default_moment_inversion_results = query_default(resultManager.moment_inversion_results)
    
    # If everything is missing, nothing to do
    if default_mcmc_results.empty and default_mle_results.empty and default_gen_results.empty and default_ift_results.empty and default_moment_inversion_results.empty:
        resultManager.console.print(f"[bold yellow]No data available for binned complex plane plots. Skipping.[/bold yellow]")
        return

    if bins_to_plot is None:
        bins_to_plot = np.arange(resultManager.n_mass_bins)

    cols_to_plot = resultManager.waveNames
    nrows, ncols = calculate_subplot_grid_size(len(cols_to_plot))
    
    saved_files = []
    if not default_mle_results.empty:
        resultManager.console.print(f"Warning: MLE error ellipses does not currently propagate errors from any reference wave rotation", style="bold yellow")
    for bin in tqdm.tqdm(bins_to_plot, disable=silence):
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
        
        mass = resultManager.masses[bin]
        binned_mcmc_samples = safe_query(default_mcmc_results, f'mass == {mass}')
        binned_mle_results  = safe_query(default_mle_results,  f'mass == {mass}')
        binned_gen_results  = safe_query(default_gen_results,  f'mass == {mass}')
        binned_ift_results  = safe_query(default_ift_results,  f'mass == {mass}')
        binned_moment_inversion_results = safe_query(default_moment_inversion_results, f'mass == {mass}')
        
        if binned_mcmc_samples.empty and binned_mle_results.empty and binned_gen_results.empty and binned_ift_results.empty and binned_moment_inversion_results.empty:
            resultManager.console.print(f"[bold yellow]No data available for bin {bin} (mass {mass}). Skipping this bin.[/bold yellow]")
            plt.close()
            continue
        
        # share limits so we know some amplitudes are small, better at identifying phase mismatches
        max_lim = -np.inf
        for _df in [binned_mcmc_samples, binned_mle_results, binned_gen_results, binned_ift_results]:
            if not _df.empty:
                all_amps = np.array([[np.real(_df[f'{wave}_amp']), np.imag(_df[f'{wave}_amp'])] for wave in cols_to_plot])
                df_max_lim = max(np.abs(all_amps.max()), np.abs(all_amps.min()))
                max_lim = max(max_lim, df_max_lim)
        bin_edges = np.linspace(-max_lim, max_lim, 100)

        # Store handles for legend
        handles = []

        for i, wave in enumerate(cols_to_plot):
            
            reference_wave = resultManager.sector_to_ref_wave[wave[-1]].strip("_amp")
            is_reference = wave == reference_wave
            
            if not binned_mcmc_samples.empty and wave in binned_mcmc_samples.columns:
                cval = binned_mcmc_samples[f"{wave}_amp"] # [chain_offset*nsamples:(chain_offset+1)*nsamples]
                rval, ival = np.real(cval), np.imag(cval)
                reference_intensity = np.array(binned_mcmc_samples[reference_wave])
                norm = plt.Normalize(reference_intensity.min(), reference_intensity.max())
                cmap = plt.cm.inferno
                        
                #### PLOT MCMC RESULTS ####
                if np.all(np.abs(ival) < 1e-5): # if imaginary part is ~ 0 then plot real part as a histogram (i.e. reference waves)
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
                    if mcmc_selection == "random":
                        rnd = np.random.permutation(np.arange(len(rval)))[:mcmc_nsamples]
                    elif mcmc_selection == "thin":
                        rnd = np.linspace(0, len(rval)-1, mcmc_nsamples, dtype=int)
                    elif mcmc_selection == "last":
                        rnd = np.arange(len(rval)-mcmc_nsamples, len(rval))
                    axes[i].scatter(rval[rnd], ival[rnd], alpha=0.1, c=reference_intensity[rnd], cmap=cmap, norm=norm)
                    axes[i].set_xlabel("Real", size=12)
                    axes[i].set_ylabel("Imaginary", size=12)
                    axes[i].set_xlim(-max_lim, max_lim)
                    axes[i].set_ylim(-max_lim, max_lim)                
                if i==0:
                    mcmc_legend_line = axes[i].plot([], [], color=mcmc_color, alpha=1.0, linewidth=2, label="mcmc")[0]
                    handles.append(mcmc_legend_line)
                    
            #### PLOT MLE RESULTS ####
            if not binned_mle_results.empty and wave in binned_mle_results.columns:
                for irow in binned_mle_results.index: # for each random start MLE fit
                    cval = binned_mle_results.loc[irow, f"{wave}_amp"]
                    real_part = np.real(cval)
                    imag_part = np.imag(cval)
                    real_part_semimajor = binned_mle_results.loc[irow, f'{wave}_re_err']
                    imag_part_semiminor = binned_mle_results.loc[irow, f'{wave}_im_err']
                    if is_reference:
                        axes[i].axvspan(real_part - real_part_semimajor, real_part + real_part_semimajor, color=mle_color, alpha=0.01)
                        axes[i].axvline(real_part, color=mle_color, linestyle='--', alpha=0.5, linewidth=1)
                    else:
                        part_angle = binned_mle_results.loc[irow, f'{wave}_err_angle'] * 180 / np.pi
                        ellipse = Ellipse(xy=(real_part, imag_part), width=2 * real_part_semimajor, height=2 * imag_part_semiminor, angle=part_angle, facecolor='none', edgecolor='xkcd:red', alpha=0.5)
                        axes[i].add_patch(ellipse)                
                if i==0:
                    mle_legend_line = axes[i].plot([], [], color=mle_color, alpha=1.0, linewidth=2, label="mle")[0]
                    handles.append(mle_legend_line)
            
            #### PLOT GENERATED RESULTS ####
            if not binned_gen_results.empty and wave in binned_gen_results.columns:
                gen_amp = binned_gen_results[f"{wave}_amp"].values[0]
                if is_reference:
                    axes[i].axvline(np.real(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=2)
                else:
                    axes[i].axhline(np.imag(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0)
                    axes[i].axvline(np.real(gen_amp), color=gen_color, linestyle='dashdot', alpha=1.0)
                if i==0:
                    gen_legend_line = axes[i].plot([], [], color=gen_color, linestyle='dashdot', alpha=1.0, linewidth=2, label="gen")[0]
                    handles.append(gen_legend_line)
                   
            #### PLOT MOMENT INVERSION RESULTS ####
            if not binned_moment_inversion_results.empty and wave in binned_moment_inversion_results.columns:
                if is_reference:
                    for irow in range(len(binned_moment_inversion_results)):
                        real_part = np.real(binned_moment_inversion_results.iloc[irow][f'{wave}_amp'])
                        axes[i].axvline(real_part, color=moment_inversion_color, linestyle='-', alpha=0.1, linewidth=0.1)
                else:
                    real_part = np.real(binned_moment_inversion_results[f'{wave}_amp'])
                    imag_part = np.imag(binned_moment_inversion_results[f'{wave}_amp'])
                    axes[i].scatter(real_part, imag_part, color=moment_inversion_color, alpha=0.4, marker='o', s=2)                
                if i==0:
                    moment_inversion_legend_line = axes[i].plot([], [], color=moment_inversion_color, alpha=1.0, linewidth=2, label="mom inv")[0]
                    handles.append(moment_inversion_legend_line)
                    
            #### PLOT NIFTY FIT RESULTS ####
            if not binned_ift_results.empty and wave in binned_ift_results.columns:
                real_part = np.real(binned_ift_results[f"{wave}_amp"])
                imag_part = np.imag(binned_ift_results[f"{wave}_amp"])
                semimajor = 0 # If we only have 1 sample -> no covariance -> no error -> no error ellipse
                semiminor = 0
                if len(real_part) > 1:
                    cov = np.cov(real_part, imag_part)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov) # eigenvalues in ascending order, eigenvectors as columns [:, i]
                    part_angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])) # atan2 of opposite / adjacent
                    semimajor = np.sqrt(eigenvalues[1]) # larger  eigenvalue
                    semiminor = np.sqrt(eigenvalues[0]) # smaller eigenvalue                    
                if is_reference:
                    if not np.allclose(imag_part, 0):
                        raise ValueError("Current implementation expects reference waves to have zero imaginary part")
                    if len(real_part) > 1:
                        axes[i].axvspan(np.mean(real_part) - semimajor, np.mean(real_part) + semimajor, color=ift_color_dict["Signal"], alpha=0.4)
                    axes[i].axvline(np.mean(real_part), color=ift_color_dict["Signal"], linestyle='--', alpha=1.0, linewidth=1)
                else:
                    if len(real_part) > 1:
                        ellipse = Ellipse(xy=(np.mean(real_part), np.mean(imag_part)), 
                                         width  = 2 * semimajor, 
                                         height = 2 * semiminor, 
                                         angle  = part_angle, 
                                         facecolor='none', edgecolor=ift_color_dict["Signal"], alpha=1.0, linewidth=2)
                        axes[i].add_patch(ellipse)
                    axes[i].scatter(real_part, imag_part, color=ift_color_dict["Signal"], alpha=0.4, marker='o', s=2)
                if i==0:
                    ift_legend_line = axes[i].plot([], [], color=ift_color_dict["Signal"], linestyle='-', alpha=1.0, linewidth=2, label="ift")[0]
                    handles.append(ift_legend_line)

            # Instead of title we add a text in the top right corner
            axes[i].text(0.975, 0.975, prettyLabels[wave],
                        size=20, color='black', fontweight='bold',
                        horizontalalignment='right', verticalalignment='top',
                        transform=axes[i].transAxes)
            
            # Add cross hairs for (0, 0)
            axes[i].axvline(0, color='black', linestyle='--', alpha=0.2, linewidth=1)
            axes[i].axhline(0, color='black', linestyle='--', alpha=0.2, linewidth=1)

        axes[0].legend(handles=handles, loc='upper left', prop={'size': 8})

        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/complex_plane/bin{bin}_complex_plane.{file_type}"
        save_and_close_fig(ofile, fig, axes, console=resultManager.console, overwrite=True, verbose=False)
        saved_files.append(ofile)
        plt.close()

    resultManager.console.print(f"\nSaved '{name}' plots to:")
    for file in saved_files:
        resultManager.console.print(f"  - {file}")
    resultManager.console.print(f"\n")
        
def plot_overview_across_bins(resultManager: ResultManager, mcmc_nsamples_per_bin=300, mcmc_selection="thin", file_type='pdf'):
    """
    This is a money plot. Two plots stacked vertically (intensity on top, relative phases on bottom)
    
    Args:
        resultManager: ResultManager instance
        mcmc_nsamples_per_bin: Number of MCMC samples to use per bin
        mcmc_selection: ["random", "thin", "last"] mcmc_nsamples will be selected by this criteria
            "random": random samples will be used
            "thin": draw mcmc_nsamples using nsamples / mcmc_nsamples steps
            "last": last mcmc_nsamples will be used
    """
    name = "intensity + phases"
    resultManager.console.print(header_fmt.format(f"Plotting '{name}' plots..."))
    
    hist_results = query_default(resultManager.hist_results)
    mcmc_results = query_default(resultManager.mcmc_results)
    mle_results  = query_default(resultManager.mle_results)
    gen_results  = query_default(resultManager.gen_results[0])
    ift_results  = query_default(resultManager.ift_results[0])
    moment_inversion_results = query_default(resultManager.moment_inversion_results)
    
    # If everything is missing, nothing to do
    if (hist_results.empty and mcmc_results.empty and mle_results.empty and 
        gen_results.empty and ift_results.empty and moment_inversion_results.empty):
        resultManager.console.print(f"[bold yellow]No data available for binned intensity plots. Skipping.[/bold yellow]")
        return

    waveNames = resultManager.waveNames
    n_mass_bins = resultManager.n_mass_bins
    massBins = resultManager.massBins
    masses = resultManager.masses
    line_half_width = resultManager.mass_bin_width / 2
    
    nEvents, nEvents_err = None, None
    if not hist_results.empty:
        nEvents = hist_results['nEvents'].values
        nEvents_err = hist_results['nEvents_err'].values

    samples_to_draw = pd.DataFrame()
    if not mcmc_results.empty:
        resultManager.console.print(f"subsampling mcmc results with '{mcmc_selection}' selection\n", style="bold yellow")
        if mcmc_selection == "random":
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.sample(n=mcmc_nsamples_per_bin, replace=False)).reset_index(drop=True)
        elif mcmc_selection == "thin":
            step_size = mcmc_results['sample'].unique().size // mcmc_nsamples_per_bin
            if step_size == 0: step_size = 1 # requested more samples than available
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.iloc[::step_size]).reset_index(drop=True)
        elif mcmc_selection == "last":
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.iloc[-mcmc_nsamples_per_bin:]).reset_index(drop=True)
    
    for k, waveName in enumerate(waveNames):
        
        reference_wave = 'Sp0+' if waveName[-1] == "+" else 'Sp0-'
        
        # Create a new figure for each waveName
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(wspace=0, hspace=0.03)
        
        # Main intensity plot
        ax = axes[0]
        ax.set_xlim(1.04, 1.72)
        
        # Track available legend elements. We primarily do this so we have control over the alpha of the
        #   legend lines. We do this by creating empty plots with appropriate styles
        error_bars = None
        mcmc_legend_line = None
        mle_bars = None
        ift_legend_lines = []
        moment_inversion_line = None
        
        #### PLOT DATA HISTOGRAM
        # ax.step(masses, nEvents[0], where='post', color='black', alpha=0.8)
        if nEvents is not None and nEvents_err is not None:
            data_line = hep.histplot((nEvents, massBins), ax=ax, color='black', alpha=0.8)
            error_bars = ax.errorbar(masses, nEvents, yerr=nEvents_err, 
                        color='black', alpha=0.8, fmt='o', markersize=2, capsize=3, label="Data")
        
        #### PLOT GENERATED CURVE
        if not gen_results.empty:
            gen_cols = [col for col in gen_results.columns if waveName in col and "_amp" not in col]
            if len(gen_cols) == 0:
                resultManager.console.print(f"[bold yellow]No generated results found for {waveName}[/bold yellow]")
            for col in gen_cols:
                if "_cf" in col: fit_type = "Bkgnd"
                elif len(col.split("_")) > 1: fit_type = "Param"
                else: fit_type = "Signal"
                ax.plot(gen_results['mass'], gen_results[col], color='white',  linestyle='-', alpha=0.4, linewidth=4, zorder=9)
                ax.plot(gen_results['mass'], gen_results[col], color=ift_color_dict[fit_type],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)

        #### PLOT MCMC FIT INTENSITY
        if not samples_to_draw.empty:
            for bin_idx in range(n_mass_bins):
                mass = masses[bin_idx]
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
        if not ift_results.empty:
            ift_cols = [col for col in ift_results.columns if waveName in col and "_amp" not in col]
            if len(ift_cols) == 0:
                resultManager.console.print(f"[bold yellow]No 'IFT' results found for {waveName}[/bold yellow]")
            ift_nsamples = ift_results['sample'].unique().size
            for col in ift_cols:
                if "_cf" in col: fit_type = "Bkgnd"
                elif len(col.split("_")) > 1: fit_type = "Param"
                else: fit_type = "Signal"
                for isample in range(ift_nsamples):
                    tmp = ift_results.query('sample == @isample')
                    ax.plot(tmp['mass'], tmp[col], color=ift_color_dict[fit_type], 
                                    linestyle='-', alpha=ift_alpha, linewidth=1, label=fit_type)
                    if isample == 0: # create empty plot just for the legend to have lines with different alpha
                        ift_line = ax.plot([], [], color=ift_color_dict[fit_type], linestyle='-', alpha=1.0, linewidth=1, label=fit_type)[0]
                        ift_legend_lines.append(ift_line)
        
        #### PLOT MOMENT INVERSION INTENSITY
        if not moment_inversion_results.empty:
            for bin_idx in range(n_mass_bins):
                mass = masses[bin_idx]
                binned_samples = moment_inversion_results.query('mass == @mass')
                x_starts = np.full_like(binned_samples[waveName], mass - line_half_width)
                x_ends   = np.full_like(binned_samples[waveName], mass + line_half_width)
                ax.hlines(y=binned_samples[waveName],
                        xmin=x_starts,
                        xmax=x_ends,
                        colors=moment_inversion_color,
                        alpha=0.05,
                        linewidth=1)
            moment_inversion_line = ax.plot([], [], color=moment_inversion_color, alpha=0.7, linewidth=2, label="Moment Inv.")[0]
        
        #### PLOT MLE FIT INTENSITY
        if not mle_results.empty:
            jitter_scale = mle_jitter_scale * resultManager.mass_bin_width
            mass_jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(mle_results)) # shared with phases below
            ax.errorbar(mle_results['mass'] + mass_jitter, mle_results[waveName], yerr=mle_results[f"{waveName}_err"], 
                        color=mle_color, alpha=0.2, fmt='o', markersize=2, capsize=3)
            mle_bars = ax.plot([], [], label="MLE", color=mle_color, alpha=1.0,  markersize=2)[0]
        
        # Create the legend with the style from iftpwa_plot.py
        handles = []
        if error_bars is not None: handles.append(error_bars)
        if mcmc_legend_line is not None: handles.append(mcmc_legend_line)
        if mle_bars is not None: handles.append(mle_bars)
        if moment_inversion_line is not None: handles.append(moment_inversion_line)
        handles += ift_legend_lines
        ax.legend(handles=handles, labelcolor="linecolor", handlelength=0.3, 
                 handletextpad=0.15, frameon=False, loc='upper right', prop={'size': 16})
        
        #################################
        ##### BEGIN PLOTTING PHASES #####
        #################################

        phase_ax = axes[1]
        phase_ax.set_ylim(-180, 180)
        phase_ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        #### PLOT MCMC PHASES
        if not samples_to_draw.empty:
            for bin_idx in range(n_mass_bins):
                mass = masses[bin_idx]
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
        
        #### PLOT GENERATED PHASES
        # For NIFTy plots we also plot the mirror ambiguity (since sometimes the fit finds the reflection of the generated phase)
        if not gen_results.empty:
            phase = np.angle(gen_results[f"{waveName}_amp"], deg=True)
            phase = np.unwrap(phase, period=360)
            for offset in [-360, 0, 360]:
                phase_ax.plot(gen_results['mass'], phase + offset, color='white',  linestyle='-', alpha=0.3, linewidth=4, zorder=9) # highlight to make more noticeable
                phase_ax.plot(gen_results['mass'], phase + offset, color=ift_color_dict["Signal"],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)
                phase_ax.plot(gen_results['mass'], -phase + offset, color='white',  linestyle='-', alpha=0.3, linewidth=4, zorder=9)
                phase_ax.plot(gen_results['mass'], -phase + offset, color=ift_color_dict["Signal"],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)
            
        #### PLOT NIFTY PHASES
        # For NIFTy plots we also plot the mirror ambiguity (since sometimes the fit finds the reflection of the generated phase)
        if not ift_results.empty:
            for isample in range(ift_nsamples):
                tmp = ift_results.query('sample == @isample')
                phase = np.angle(tmp[f"{waveName}_amp"], deg=True)
                phase = np.unwrap(phase, period=360)
                for offset in [-360, 0, 360]:
                    phase_ax.plot(tmp['mass'],  phase + offset, color=ift_color_dict["Signal"], linestyle='-', alpha=ift_alpha, linewidth=1)
                    phase_ax.plot(tmp['mass'], -phase + offset, color=ift_color_dict["Signal"], linestyle='-', alpha=ift_alpha, linewidth=1)
            
        #### PLOT MOMENT INVERSION PHASES
        if not moment_inversion_results.empty and f"{waveName}_amp" in moment_inversion_results.columns:
            for bin_idx in range(n_mass_bins):
                mass = masses[bin_idx]
                binned_samples = moment_inversion_results.query('mass == @mass')
                x_starts = np.full_like(binned_samples[waveName], mass - line_half_width)
                x_ends   = np.full_like(binned_samples[waveName], mass + line_half_width)
                phases = np.angle(binned_samples[f"{waveName}_amp"], deg=True)
                phase_ax.hlines(y=phases,   
                        xmin=x_starts, 
                        xmax=x_ends, 
                        colors=moment_inversion_color, 
                        alpha=0.1, 
                        linewidth=1)
            
        #### PLOT MLE PHASES
        if not mle_results.empty:
            phase = mle_results[f"{waveName}_relative_phase"]
            phase_error = mle_results[f"{waveName}_relative_phase_err"]
            phase_ax.errorbar(mle_results['mass'] + mass_jitter, phase, yerr=phase_error, 
                            color=mle_color, alpha=0.2, fmt='o', markersize=2, capsize=3)
            
        #### PLOT WAVE NAME IN CORNER
        text = ax.text(0.05, 0.92, f"{prettyLabels[waveName]}", transform=ax.transAxes, fontsize=36, c='black', zorder=9)
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        
        #### AXIS SETTINGS
        ax.set_ylim(0, 1.15 * np.max(nEvents))
        phase_ax.set_xlabel(r"$m_X$ [GeV]", size=20)
        phase_ax.tick_params(axis='x', labelsize=16)
        ax.set_ylabel(rf"Intensity / {resultManager.mass_bin_width:.3f} GeV", size=22)
        ax.tick_params(axis='y', labelsize=16)
        wave_label1 = prettyLabels[waveName].strip("$")
        wave_label2 = prettyLabels[reference_wave].strip("$")
        phase_ax.set_ylabel(f"$\phi_{{{wave_label1}}} - \phi_{{{wave_label2}}}$ [deg]", size=18)

        plt.tight_layout()
        ofile = f"{resultManager.base_directory}/{default_plot_subdir}/intensity_and_phases/intensity_phase_plot_{waveName}.{file_type}"
        save_and_close_fig(ofile, fig, axes, console=resultManager.console, overwrite=True, verbose=True)
        plt.close()
        
    resultManager.console.print(f"\n")
    
def plot_moments_across_bins(resultManager: ResultManager, mcmc_nsamples_per_bin=300, mcmc_selection="thin", file_type='pdf'):
    
    """
    This is another money plot. All non-zero projected moments are plotted
    
    Args:
        resultManager: ResultManager instance
        mcmc_nsamples_per_bin: Number of MCMC samples to use per bin
        mcmc_selection: ["random", "thin", "last"] mcmc_nsamples will be selected by this criteria
            "random": random samples will be used
            "thin": draw mcmc_nsamples using nsamples / mcmc_nsamples steps
            "last": last mcmc_nsamples will be used
    """
    name = "moments"
    resultManager.console.print(header_fmt.format(f"Plotting '{name}' plots..."))
    
    if resultManager.moment_latex_dict is None:
        resultManager.console.print("[bold yellow]Does not appear that moments have been calculated yet. Skipping.[/bold yellow]")
        return
    
    mcmc_results = query_default(resultManager.mcmc_results)
    mle_results  = query_default(resultManager.mle_results)
    gen_results  = query_default(resultManager.gen_results[0])
    ift_results  = query_default(resultManager.ift_results[0])
    
    # If everything is missing, nothing to do
    if mcmc_results.empty and mle_results.empty and gen_results.empty and ift_results.empty:
        resultManager.console.print(f"[bold yellow]No data available for binned intensity plots. Skipping.[/bold yellow]")
        return
    
    n_mass_bins = resultManager.n_mass_bins
    masses = resultManager.masses
    line_half_width = resultManager.mass_bin_width / 2
    
    samples_to_draw = pd.DataFrame()
    if not mcmc_results.empty:
        if mcmc_selection == "random":
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.sample(n=mcmc_nsamples_per_bin, replace=False)).reset_index(drop=True)
        elif mcmc_selection == "thin":
            step_size = mcmc_results['sample'].unique().size // mcmc_nsamples_per_bin
            if step_size == 0: step_size = 1 # requested more samples than available
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.iloc[::step_size]).reset_index(drop=True)
        elif mcmc_selection == "last":
            samples_to_draw = mcmc_results.groupby('mass')[mcmc_results.columns].apply(lambda x: x.iloc[-mcmc_nsamples_per_bin:]).reset_index(drop=True)

    def plot_moment(moment_name, ofile):
        fig, axis = plt.subplots(figsize=(10, 10))
        axis.set_ylabel(f"Moment Value / {resultManager.mass_bin_width:.3f} GeV$/c^2$", size=22)
        axis.set_xlabel("$m_X$ [GeV$/c^2$]", size=22)
        
        # Track available legend elements. We primarily do this so we have control over the alpha of the
        #   legend lines. We do this by creating empty plots with appropriate styles
        mcmc_legend_line = None
        mle_bars = None
        ift_legend_lines = []
        
        # NOTE: Assume that the set of moments are all the same in the DataFrames as they should all have the same waveset
        moment_is_zero = False
        dfs_to_process = [
            ("mle",  mle_results), 
            ("mcmc", mcmc_results),
            ("ift",  ift_results),
            ("gen",  gen_results)
        ]
        for df_name, _df in dfs_to_process:
            if not _df.empty:
                if moment_name not in _df.columns:
                    resultManager.console.print(f"[bold yellow]Moment '{moment_name}' not found in DataFrame '{df_name}', skipping.[/bold yellow]")
                    plt.close()
                    return
                moment_value = _df[moment_name].values # no need to rescale with bpg (bins_per_group). Zero is zero
                if np.allclose(moment_value, 0, atol=1e-6):
                    moment_is_zero = True
            if moment_is_zero:
                # If moment is zero in one dataframe then skip this plot. Presumably we should just compare approaches with the same initial model?
                resultManager.console.print(f"[bold yellow]Moment '{moment_name}' is zero in DataFrame '{df_name}', skipping.[/bold yellow]")
                plt.close()
                return
        
        ### PLOT THE GENERATED MOMENTS
        maxy = -np.inf
        if not gen_results.empty:
            for sample in gen_results['sample'].unique(): # should just be 1 sample
                df_sample = gen_results.query(f'sample == {sample}')
                moment_value = df_sample[moment_name].values
                moment_value = moment_value * resultManager.bpg # ift generally uses finer binning than amptools, rescale to match
                axis.plot(df_sample['mass'], moment_value, color='white',  linestyle='-', alpha=0.4, linewidth=4, zorder=9)
                axis.plot(df_sample['mass'], moment_value, color=ift_color_dict["Signal"],  linestyle='--', alpha=1.0, linewidth=3, zorder=10)
                maxy = np.max([maxy, np.max(moment_value)])
            
        #### PLOT MCMC FIT INTENSITY
        if not samples_to_draw.empty:
            for bin_idx in range(n_mass_bins):
                mass = masses[bin_idx]
                binned_samples = samples_to_draw.query('mass == @mass')
                x_starts = np.full_like(binned_samples[moment_name], mass - line_half_width)
                x_ends   = np.full_like(binned_samples[moment_name], mass + line_half_width)
                axis.hlines(y=binned_samples[moment_name],   
                        xmin=x_starts,
                        xmax=x_ends,
                        colors=mcmc_color,
                        alpha=0.1,
                        linewidth=1)
                maxy = np.max([maxy, np.max(binned_samples[moment_name])])
            mcmc_legend_line = axis.plot([], [], color=mcmc_color, alpha=0.7, linewidth=2, label="MCMC")[0]
        
        #### PLOT IFT MOMENTS
        if not ift_results.empty:
            for isample, sample in enumerate(ift_results['sample'].unique()):
                df_sample = ift_results.query('sample == @sample')
                axis.plot(df_sample['mass'], df_sample[moment_name], color=ift_color_dict["Signal"], 
                                linestyle='-', alpha=ift_alpha, linewidth=1)
                if isample == 0: # create empty plot just for the legend to have lines with different alpha
                    ift_line = axis.plot([], [], color=ift_color_dict["Signal"], linestyle='-', alpha=1.0, linewidth=1, label="IFT")[0]
                    ift_legend_lines.append(ift_line)
                maxy = np.max([maxy, np.max(df_sample[moment_name])])
            
        #### PLOT MLE MOMENTS
        if not mle_results.empty:
            jitter_scale = mle_jitter_scale * resultManager.mass_bin_width
            mass_jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(mle_results)) # shared with phases below
            axis.errorbar(mle_results['mass'] + mass_jitter, mle_results[moment_name], yerr=0, 
                        color=mle_color, alpha=0.2, fmt='o', markersize=3, capsize=0)
            mle_bars = axis.plot([], [], label="MLE", color=mle_color, alpha=1.0,  markersize=2)[0]
            maxy = np.max([maxy, np.max(mle_results[moment_name])])

        handles = []
        if mcmc_legend_line is not None: handles.append(mcmc_legend_line)
        if mle_bars is not None: handles.append(mle_bars)
        handles += ift_legend_lines
        axis.legend(handles=handles, labelcolor="linecolor", handlelength=0.3,
                 handletextpad=0.15, frameon=False, loc='upper right', prop={'size': 16})
        
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        text = axis.text(0.05, 0.92, f"{resultManager.moment_latex_dict[moment_name]}", transform=axis.transAxes, fontsize=36, c='black', zorder=9)
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        axis.set_box_aspect(1.0)
        axis.tick_params(axis='x', labelsize=16)
        axis.tick_params(axis='y', labelsize=16)
        miny = axis.get_ylim()[0]
        yrange = maxy - miny
        axis.set_ylim(miny, miny + 1.15 * yrange) # leave minimum the same, increase height so we can squeeze in some text at the top
        plt.tight_layout()
        
        save_and_close_fig(ofile, fig, axis, console=resultManager.console, overwrite=True, verbose=True)

    for moment_name in resultManager.moment_latex_dict.keys():
        plot_moment(moment_name, ofile=f"{resultManager.base_directory}/{default_plot_subdir}/moments/moment_{moment_name}.{file_type}")
    
def montage_and_gif_select_plots(resultManager: ResultManager, file_type='pdf'):
    
    resultManager.console.print(header_fmt.format(f"Montaging / GIFing all plots..."))
    
    base_directory = resultManager.base_directory
    
    subdirs = ["complex_plane", "intensity"]
    for subdir in subdirs:
        output_directory = f"{base_directory}/{default_plot_subdir}/{subdir}"
        
        # Sort files by bin number
        bin_files_path = f"{output_directory}/bin*.{file_type}"
        bin_files = sorted(glob.glob(bin_files_path), 
                          key=lambda x: int(re.search(r'bin(\d+)', x).group(1)))
        if bin_files:
            files_str = " ".join(bin_files)            
            montage_output = f"{output_directory}/montage_output.{file_type}"
            gif_output = f"{output_directory}/output.gif"
            resultManager.console.print(f"Create montage + gif of plots in '{output_directory}'")
            os.system(f"montage {files_str} -density 300 -geometry +10+10 {montage_output}")
            os.system(f"convert -delay 40 {files_str} -layers optimize -colors 256 -fuzz 2% {gif_output}")
        else:
            resultManager.console.print(f"[bold yellow]No bin files found in {bin_files_path}[/bold yellow]")
        
    subdirs = ["intensity_and_phases"]
    for subdir in subdirs:
        output_directory = f"{base_directory}/{default_plot_subdir}/{subdir}"
        files = [] # Montage in the same order as waveNames
        for wave in resultManager.waveNames:
            files_path = f"{output_directory}/intensity_phase_plot_{wave}.{file_type}"
            files.append(files_path)
        if files:
            files_str = " ".join(files)
            montage_output = f"{base_directory}/{default_plot_subdir}/{subdir}/montage_output.{file_type}"
            resultManager.console.print(f"Create montage of plots in '{base_directory}/{default_plot_subdir}/{subdir}'")
            os.system(f"montage {files_str} -density 300 -geometry +10+10 {montage_output}")
        else:
            resultManager.console.print(f"[bold yellow]No intensity phase plot files found in {output_directory}[/bold yellow]")
            
    subdirs = ["moments"]
    for subdir in subdirs:
        output_directory = f"{base_directory}/{default_plot_subdir}/{subdir}"
        files = glob.glob(f"{output_directory}/moment_*.{file_type}")
        files = [f.replace('(', '\(').replace(')', '\)') for f in files]
        files = sorted(files)
        if files:
            files_str = " ".join(files)
            montage_output = f"{output_directory}/montage_output.{file_type}"
            resultManager.console.print(f"Create montage of plots in '{output_directory}'")
            os.system(f"montage {files_str} -density 300 -geometry +10+10 {montage_output}")
    
    resultManager.console.print(f"\n")
