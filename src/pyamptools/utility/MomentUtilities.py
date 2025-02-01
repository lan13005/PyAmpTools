from pyamptools.utility.general import converter, channel_map
from pyamptools.utility.MomentCalculator import (
    AmplitudeSet,
    AmplitudeValue,
    QnWaveIndex,
)
import pandas as pd
from typing import List
import numpy as np
from multiprocessing import Pool

# This file contains additional code acting as an interface between Boris's MomentCalculator and PyAmpTools

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
        
        ls = []
        channels = []
        for w in wave_names:
            ls.append(converter[w][1])
            channels.append(channel_map[w])
        
        if len(set(channels)) > 1:
            # NOTE: One scenario this could happen is if one uses (for instance) an Isotropic wave.
            #       I am currently unsure how to handle this case, so crash the program for now
            raise ValueError("MomentManager| Multiple channels found in wave_names. This is not supported!")
        if channels[0] != 'TwoPseudoscalar':
            raise ValueError("MomentManager| Only the two pseudoscalar system is supported at the moment!")
        
        lmax = max(ls)
        self.max_L = 2 * lmax
        
        print(f"MomentManager| Calculating moments assuming a {channels[0]} system with max L = {self.max_L}")

    def process_and_return_df(self, normalization_scheme=0, pool_size=4, append=True):
        
        """
        Process the partial wave amplitudes DataFrame and return a new dataframe with the moments
        
        Args:
            normalization_scheme (int): Normalization scheme to use: {0: normalize to H_0(0, 0), 1: norm to acceptance corrected intensity 2: normalize to uncorrected intensity in the bin}
            pool_size (int): Number of processes to use by multiprocessing.Pool
            append (bool): Whether to append the moments dataframe to the original dataframe
            
        Returns:
            moment_results (pd.DataFrame): DataFrame containing (atleast) the moments
            latex_name_dict (dict): Dictionary with the latex naming of the moments
        """

        with Pool(pool_size) as pool:
            moment_results = pool.starmap(self.calc_moments, [(i, normalization_scheme) for i in range(len(self.df))])
            
        # Convert to list of dicts to dict of lists then into a dataframe
        moment_results = {k: [d[k] for d in moment_results] for k in moment_results[0].keys()}
        moment_results = pd.DataFrame(moment_results, index=self.df.index)
        moment_results = moment_results.drop(columns=['index'])
        
        # Perform symmetry checks and gather latex naming
        moment_names = [col for col in moment_results.columns if col.startswith("H")]
        latex_name_dict = {}
        for moment_name in moment_names:
            values, part_taken = symmetry_check(moment_results[moment_name])
            moment_results[moment_name] = values
            i, L, M = moment_name[1], moment_name[3], moment_name[5]
            latex_name_dict[moment_name] = rf"$\{part_taken}[H_{{{i}}}({L}, {M})]$"

        if append: moment_results = pd.concat([self.df, moment_results], axis=1)
        
        return moment_results, latex_name_dict
        
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