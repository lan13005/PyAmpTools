from pyamptools.utility.io import loadAmpToolsResultsFromYaml
from pyamptools.utility.general import converter
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import os
import argparse
from pyamptools.utility.general import Timer

from multiprocessing import Pool

from pyamptools.utility.MomentCalculator import (
    AmplitudeSet,
    AmplitudeValue,
    QnWaveIndex,
)

mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
    'axes.grid': True,
    'grid.alpha': 0.7,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

def read_partial_wave_amplitudes(amp_series: pd.Series, wave_names: List[str]) -> List[AmplitudeValue]:
    
    """
    Reads partial-wave amplitudes values for given mass bin from DataFrame and returns a list of AmplitudeValue objects
        usable by MomentCalculator
    """

    partial_wave_amplitudes = []
    for wave_name in wave_names:
        refl, l, m = converter[wave_name]
        partial_wave_amplitudes.append(AmplitudeValue(QnWaveIndex(refl=refl, l=l, m=m), val=amp_series[f"{wave_name}_amp"]))
    
    return partial_wave_amplitudes

class MomentManager:
    
    def __init__(self, df, wave_names):
        
        """
        Manages moment calculations across a dataframe of complex partial wave amplitudes with various normalization options
        
        Args:
            df (pd.DataFrame): DataFrame of complex partial wave amplitudes
            wave_names (List[str]): List of wave names
            mass_centers (np.ndarray, None): Array of mass centers
            tprime_centers (np.ndarray, None): Array of tprime centers
        """
        
        self.df = df
        self.wave_names = wave_names
        
        lmax = max([converter[w][1] for w in wave_names])
        self.max_L = 2 * lmax

    def process_and_return_df(self, normalization_scheme=0, pool_size=4, append=True):
        
        """
        Process the partial wave amplitudes DataFrame and return a new dataframe with the moments
        """
        
        from multiprocessing import Pool

        with Pool(pool_size) as pool:
            moment_results = pool.starmap(self.calc_moments, [(i, normalization_scheme) for i in range(len(self.df))])
            
        # Convert to list of dicts to dict of lists then into a dataframe
        moment_results = {k: [d[k] for d in moment_results] for k in moment_results[0].keys()}
        moment_results = pd.DataFrame(moment_results, index=self.df.index)
        moment_results = moment_results.drop(columns=['index'])

        if append: moment_results = pd.concat([self.df, moment_results], axis=1)
        
        return moment_results
        
    def calc_moments(self, i, normalization_scheme=0):
        
        """
        0: normalize to H_0(0, 0)
        1: normalize to the acceptance corrected intensity in the bin
        2: normalize to the uncorrected intensity in the bin (unsure if this makes sense but will allow it)
        """
        
        moment_results = {}

        if normalization_scheme == 0:
            normalization = True
        elif normalization_scheme == 1:
            normalization = int(self.df.iloc[i]['intensity_corr'].real)
        elif normalization_scheme == 2:
            normalization = int(self.df.iloc[i]['intensity'].real)
        else:
            raise ValueError(f"Invalid normalization scheme: {normalization_scheme}")
        
        amplitudes = read_partial_wave_amplitudes(self.df.iloc[i], self.wave_names)
        amplitude_set = AmplitudeSet(amps=amplitudes, tolerance=1e-10)
        moment_result = amplitude_set.photoProdMomentSet(
            maxL=self.max_L,
            normalize=normalization,
            printMomentFormulas=False
        )
        
        moment_results['index'] = self.df.iloc[i].name # name attribute is the actual index
        for moment in moment_result.values:
            moment_results[moment.qn.label] = moment.val

        return moment_results

def symmetry_check(moment_values):
    if isinstance(moment_values, pd.Series):
        moment_values = moment_values.values
    if np.all(np.isclose(moment_values.real, 0)):
        moment_values = moment_values.imag
        part_taken = "Im"
    elif np.all(np.isclose(moment_values.imag, 0)):
        moment_values = moment_values.real
        part_taken = "Re"
    elif np.all(np.isclose(moment_values.real, 0)) and np.all(np.isclose(moment_values.imag, 0)):
        moment_values = np.zeros_like(moment_values)
    else:
        raise ValueError("Plotting code expects moments to be either real or imaginary at the moment")
    return moment_values, part_taken

def plot_moment(moment_name, t, amptools_df, ift_df, show_samples=True, no_errorbands=False, save_file=None):

    fig, ax_intens = plt.subplots()
    
    amptools_df = amptools_df.query(f'tprime == {t}')
    ift_df = ift_df.query(f'tprime == {t}')
    
    amptools_masses = sorted(amptools_df['mass'].unique())
    mass_width = np.round(amptools_masses[1] - amptools_masses[0], 4)

    ### SET THE STYLE
    ax_intens.set_title(latex_name_dict[moment_name], loc="right")
    ax_intens.set_box_aspect(1.0)
    ax_intens.tick_params(direction="in", which="both")
    ax_intens.minorticks_on()
    ax_intens.set_ylabel(f"Moment Value / {mass_width} GeV$/c^2$")
    ax_intens.ticklabel_format(axis="y", style="sci", scilimits=(-1, 3))
    ax_intens.set_xlabel("$m_X$ [GeV$/c^2$]")
    
    ## PLOT THE IFT RESULT
    if show_samples:
        for sample in ift_df['sample'].unique():
            df_sample = ift_df.query(f'sample == {sample}')
            moment_value = df_sample[moment_name].values
            moment_value = moment_value * bpg # ift generally uses finer binning than amptools, rescale to match
            label = "IFT Result" if sample == ift_df['sample'].unique()[0] else None
            ax_intens.plot(df_sample['mass'], moment_value, color="xkcd:sea blue", alpha=0.2, zorder=0, label=label)
        
    if not no_errorbands:
        mean = ift_mom_df.groupby(['tprime', 'mass'])[moment_name].mean().reset_index()
        std  = ift_mom_df.groupby(['tprime', 'mass'])[moment_name].std().reset_index()
        moment_value_low  = (mean[moment_name] - std[moment_name]) * bpg
        moment_value_high = (mean[moment_name] + std[moment_name]) * bpg
        label = label if show_samples else "IFT Result"
        ax_intens.fill_between(mass_centers, moment_value_low, moment_value_high, color="xkcd:sea blue", alpha=0.2, label=label)
        
    ## PLOT THE AMPTOOLS RESULT
    moment_value = amptools_df[moment_name].values
    ax_intens.scatter(amptools_df['mass'], moment_value, color="black", marker="o", s=20, label='Mass Indep. Result')
    ax_intens.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_intens.legend(labelcolor="linecolor", handlelength=0.2, handletextpad=0.2, frameon=False)
    
    if save_file:
        print(f"momentPlotter| Saving Figure: {save_file}")
        fig.savefig(save_file, bbox_inches="tight")

    plt.close(fig)
    del fig, ax_intens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform maximum likelihood fits (using AmpTools) on all cfg files")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("-d", type=str, default='null', help="Dump log files for each bin to this relative path. If empty str will dump to stdout")
    args = parser.parse_args()
    yaml_name = args.yaml_name
    dump = args.d

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    timer = Timer()
    
    ###############################
    #### LOAD AMPTOOLS RESULTS
    ###############################
    print("momentPlotter| Loading AmpTools results...")
    yaml_primary = OmegaConf.load(yaml_name)
    yaml_secondary = yaml_primary['nifty']['yaml']
    yaml_secondary = OmegaConf.load(yaml_secondary)
    amptools_df, (masses, tPrimeBins, bpg) = loadAmpToolsResultsFromYaml(yaml_primary)

    ###############################
    #### LOAD NIFTY RESULTS
    ###############################
    print("momentPlotter| Loading NIFTY results...")
    try: 
        yaml_secondary['GENERAL']['outputFolder']
    except MissingMandatoryValue:
        outputFolder = yaml_primary['nifty']['output_directory']
        yaml_secondary['GENERAL']['outputFolder'] = outputFolder
    nifty_pkl = yaml_secondary['GENERAL']['fitResultPath']

    with open(nifty_pkl, "rb") as f:
        data = pkl.load(f)
        
    wave_names = data['pwa_manager_base_information']['wave_names']
    mass_bins = data['pwa_manager_base_information']['mass_bins']
    tprime_bins = data['pwa_manager_base_information']['tprime_bins']
    mass_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
    tprime_centers = (tprime_bins[:-1] + tprime_bins[1:]) / 2
    nmb_waves = len(wave_names)
    nmb_masses = len(mass_centers)
    nmb_tprimes = len(tprime_centers)

    # Get complex signal field values
    signal_field_sample_values = data['signal_field_sample_values'] # ~ (nmb_samples, 2*n_waves, n_masses, n_tprime)
    signal_field_sample_values = signal_field_sample_values[:, :2*nmb_waves:2, :, :] + 1j * signal_field_sample_values[:, 1:2*nmb_waves:2, :, :]

    # Get number of samples
    nmb_samples = signal_field_sample_values.shape[0]

    # Reshape to flatten across samples, masses, tprime
    signal_field_sample_values = np.transpose(signal_field_sample_values, (0, 2, 3, 1)) # -> (nmb_samples, n_masses, n_tprime, n_waves)
    signal_field_sample_values = signal_field_sample_values.reshape(-1, nmb_waves) # -> (nmb_samples * n_masses * n_tprime, n_waves)

    # Create arrays for sample, mass, tprime indices
    sample_indices = np.repeat(np.arange(nmb_samples), nmb_masses * nmb_tprimes)
    mass_indices = np.tile(np.repeat(mass_centers, nmb_tprimes), nmb_samples)
    tprime_indices = np.tile(tprime_centers, nmb_samples * nmb_masses)

    # Stack all columns together
    ift_df = np.column_stack((sample_indices, mass_indices, tprime_indices, signal_field_sample_values))

    cols = ['sample', 'mass', 'tprime'] + [f'{wave_name}_amp' for wave_name in wave_names]
    ift_df = pd.DataFrame(ift_df, columns=cols)

    # Reorder columns
    cols = ['tprime', 'mass', 'sample'] + [f'{wave_name}_amp' for wave_name in wave_names]
    ift_df = ift_df[cols]

    # sample column is int, mass as float, tprime as float, wave_name_amp as complex
    ift_df['sample'] = ift_df['sample'].values.real.astype(int)
    ift_df['mass'] = ift_df['mass'].values.real.astype(float)
    ift_df['tprime'] = ift_df['tprime'].values.real.astype(float)
    ift_df[cols[3:]] = ift_df[cols[3:]].astype(complex)

    # Get the (fitted) total intensity
    intensity_samples = np.array(data['expected_nmb_events'], dtype=float) # ~ (nmb_samples, nmb_masses, nmb_tprimes)
    intensity_samples_no_acc = np.array(data['expected_nmb_events_no_acc'], dtype=float) # ~ (nmb_samples, nmb_masses, nmb_tprimes)
    intensity_samples = intensity_samples.reshape(-1, 1)
    intensity_samples_no_acc = intensity_samples_no_acc.reshape(-1, 1)
    ift_df['intensity'] = intensity_samples
    ift_df['intensity_corr'] = intensity_samples_no_acc

    # Perform check
    if not set(ift_df.columns.drop('sample')) <= set(amptools_df.columns):
        # print the missing columns in amptools_df that is in ift_df
        missing_columns = set(ift_df.columns.drop('sample')) - set(amptools_df.columns)
        raise ValueError(f"IFT and AmpTools columns do not match. Missing columns: {missing_columns}")

    ###########################################
    #### PROCESS THE AMPTOOLS AND IFT RESULTS
    ###########################################
    print("momentPlotter| Processing AmpTools and IFT results...")
    amptools_momentMan = MomentManager(amptools_df, wave_names)
    amptools_mom_df = amptools_momentMan.process_and_return_df(normalization_scheme=1, pool_size=10)

    ift_momentMan = MomentManager(ift_df, wave_names)
    ift_mom_df = ift_momentMan.process_and_return_df(normalization_scheme=1, pool_size=10)


    ####################################################
    #### Perform symmetry checks and gather latex naming
    ####################################################
    print("momentPlotter| Performing symmetry checks...")
    moment_names = [col for col in amptools_mom_df.columns if col.startswith("H")]

    latex_name_dict = {}
    for moment_name in moment_names:
        if moment_name in amptools_mom_df.columns:
            values, part_taken = symmetry_check(amptools_mom_df[moment_name])
            amptools_mom_df[moment_name] = values
            i, L, M = moment_name[1], moment_name[3], moment_name[5]
            latex_name_dict[moment_name] = rf"$\{part_taken}[H_{{{i}}}({L}, {M})]$"

    for moment_name in moment_names:
        if moment_name in ift_mom_df.columns:
            values, part_taken = symmetry_check(ift_mom_df[moment_name])
            ift_mom_df[moment_name] = values
            i, L, M = moment_name[1], moment_name[3], moment_name[5]
            assert latex_name_dict[moment_name] == rf"$\{part_taken}[H_{{{i}}}({L}, {M})]$"
            
    ####################################################
    #### Plot the moments
    ####################################################
    print("momentPlotter| Plotting moments...")
    outputFolder = yaml_secondary['GENERAL']['outputFolder']
    outputFolder = os.path.join(outputFolder, "plots/moments")
    os.makedirs(outputFolder, exist_ok=True)

    for t in tprime_centers:
        for moment_name in moment_names:
            plot_moment(moment_name, t, amptools_mom_df, ift_mom_df, save_file=f"{outputFolder}/{moment_name}_t{t}.pdf")

    print(f"momentPlotter| Elapsed time {timer.read()[2]}\n\n")
