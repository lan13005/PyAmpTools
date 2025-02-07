from pyamptools.utility.MomentCalculatorTwoPS import (
    AmplitudeSet,
    AmplitudeValue,
    QnWaveIndex,
)
from pyamptools.utility.MomentCalculatorVecPS import Wave, calculate_moment
from pyamptools.utility.general import converter, identify_channel

import pandas as pd
from typing import List, Tuple
import numpy as np
from multiprocessing import Pool
import itertools

# This file contains additional code acting as an interface
#   Both approaches has a function to calculate moments from some struct per fit result
# 1. Boris's MomentCalculatorTwoPS and PyAmpTools
# 2. Kevin's MomentCalculatorVecPS and PyAmpTools

# Base class for all MomentManager classes
class MomentManager: # ~base class
    def __init__(self, df, wave_names):   
        """
        Manages moment calculations across a dataframe of complex partial wave amplitudes with various normalization options
        
        Args:
            df (pd.DataFrame): DataFrame of complex partial wave amplitudes
            wave_names (List[str]): List of wave names
        """
        self.df = df
        self.wave_names = wave_names
        self.channel = identify_channel(wave_names)
    
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
        pass

class MomentManagerVecPS(MomentManager):
    
    def __init__(self, df, wave_names):
        super().__init__(df, wave_names)
                
        if self.channel != "VectorPseudoscalar":
            raise ValueError("MomentManagerVecPS| Only the vector-pseudoscalar system is supported for!")
        
        Js, Ms = [], []
        for w in wave_names:
            if self.channel == "VectorPseudoscalar":
                J, M = converter[w][3], converter[w][2]
            Js.append(J)
            Ms.append(M)
        self.max_J = 2 * max(Js)
        self.max_M = max(Ms)
        
        print(f"MomentManagerVecPS| Calculating moments assuming a {self.channel} system with max J = {self.max_J} and max M = {self.max_M}")
        self.coefficient_dict = {}

        # ===PREPARE quantum numbers of moments===
        self.Jv_array = np.array([0, 2])  # CG coefficient always makes Jv=1 be 0
        self.Lambda_array = np.arange(0, 3) # (-) Lambda values are directly proportional to (+) ones, no need to calculate
        self.J_array = np.arange(0, self.max_J + 1)
        self.M_array = np.arange(0, self.max_M + 1)  # like Lambda, -m ∝ +m moments

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
        moment_results = pd.DataFrame(moment_results)
        moment_results.index = moment_results['index']
        moment_results = moment_results.drop(columns=['index'])
        
        # # Perform symmetry checks and gather latex naming
        moment_names = [col for col in moment_results.columns if col.startswith("H")]
        latex_name_dict = {}
        for moment_name in moment_names:
            values, part_taken = symmetry_check(moment_results[moment_name])
            moment_results[moment_name] = values
            alpha, Jv, Lambda, J, M = moment_name[1], moment_name[3], moment_name[5], moment_name[7], moment_name[9]
            latex_name_dict[moment_name] = rf"$\{part_taken}[H_{{{alpha}}}({Jv}, {Lambda}, {J}, {M})]$"

        if append: moment_results = pd.concat([self.df, moment_results], axis=1)
        
        return moment_results, latex_name_dict

    def calc_moments(self, i, normalization_scheme=0):
        
        if normalization_scheme == 0:
            normalization = True
        elif normalization_scheme == 1:
            normalization = int(self.df.iloc[i]['intensity_corr'].real)
        elif normalization_scheme == 2:
            normalization = int(self.df.iloc[i]['intensity'].real)
        else:
            raise ValueError(f"Invalid normalization scheme: {normalization_scheme}")
        
        amplitudes = {}
        for amp in self.wave_names:
            reflectivity, spin, parity, m, l = parse_amplitude_vecps(amp)
            complex_amp = self.df.iloc[i][f'{amp}_amp']
            real_amp = complex_amp.real
            imag_amp = complex_amp.imag
            amplitudes[amp] = Wave(
                name=amp,
                reflectivity=reflectivity,
                spin=spin,
                parity=parity,
                m=m,
                l=l,
                real=real_amp,
                imaginary=imag_amp,
                scale=1,
            )
        amplitudes = tuple(amplitudes.values())
         
        # ===CALCULATE each moment and what production coefficients contribute to it===
        moment_results = {}
        # table_dict = {}
        for alpha in range(3):
            for Jv, Lambda, J, M in itertools.product(
                self.Jv_array, self.Lambda_array, self.J_array, self.M_array
            ):
                moment_str = f"H{alpha}({Jv},{Lambda},{J},{M})"

                self.coefficient_dict.clear()
                moment_val = calculate_moment(
                    alpha, Jv, Lambda, J, M, amplitudes, self.coefficient_dict
                )
                # save the results for this moment. The dictionary is copied to avoid
                # reference issues, and converted back to a python dict so it is pickleable
                # for the parallel processing
                # table_dict[moment_str] = dict(self.coefficient_dict.copy())
                if moment_str == "H0(0,0,0,0)":
                    if isinstance(normalization, bool):
                        normalization = moment_val.real
                    elif isinstance(normalization, (int, float)):
                        normalization = moment_val.real / normalization
                moment_results[moment_str] = moment_val / normalization
                
        moment_results['index'] = self.df.iloc[i].name # name attribute is the actual index
        
        return moment_results
    
class MomentManagerTwoPS(MomentManager):
    
    def __init__(self, df, wave_names):
        super().__init__(df, wave_names)
        
        if self.channel != "TwoPseudoscalar":
            raise ValueError("MomentManagerTwoPS| Only the two pseudoscalar system is supported for!")
        
        Js = []
        for w in wave_names:
            if self.channel == "TwoPseudoscalar":
                J = converter[w][1]
            Js.append(J)
        self.max_J = 2 * max(Js)     
        
        print(f"MomentManagerTwoPS| Calculating moments assuming a {self.channel} system with max J = {self.max_J}")

    def process_and_return_df(self, normalization_scheme=0, pool_size=4, append=True):

        with Pool(pool_size) as pool:
            moment_results = pool.starmap(self.calc_moments, [(i, normalization_scheme) for i in range(len(self.df))])

        # Convert to list of dicts to dict of lists then into a dataframe
        moment_results = {k: [d[k] for d in moment_results] for k in moment_results[0].keys()}
        moment_results = pd.DataFrame(moment_results, index=self.df.index)
        moment_results = moment_results.drop(columns=['index'])
        
        # Boris uses notation Hi_L_M. I like Kevin's notation for VecPS better Hi(L,M)
        # Also, Perform symmetry checks and gather latex naming
        moment_names = [col for col in moment_results.columns if col.startswith("H")]
        replace_dict = {}
        latex_name_dict = {}
        for old_name in moment_names:
            values, part_taken = symmetry_check(moment_results[old_name])
            moment_results[old_name] = values
            i, L, M = old_name[1], old_name[3], old_name[5]
            new_name = f"H{i}({L},{M})"
            replace_dict[old_name] = new_name
            latex_name_dict[new_name] = rf"$\{part_taken}[H_{{{i}}}({L}, {M})]$"
        moment_results.rename(columns=replace_dict, inplace=True)

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
        
        amplitudes = read_partial_wave_amplitudes_twops(self.df.iloc[i], self.wave_names)
        amplitude_set = AmplitudeSet(amps=amplitudes, tolerance=1e-10)
        moment_result = amplitude_set.photoProdMomentSet(
            maxL=self.max_J,
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

def read_partial_wave_amplitudes_twops(amp_series: pd.Series, wave_names: List[str]) -> List[AmplitudeValue]:
    
    """
    Reads partial-wave amplitudes values for given mass bin from DataFrame and returns a list of AmplitudeValue objects
        usable by MomentCalculatorTwoPS
    """

    partial_wave_amplitudes = []
    for wave_name in wave_names:
        refl, l, m = converter[wave_name]
        partial_wave_amplitudes.append(AmplitudeValue(QnWaveIndex(refl=refl, l=l, m=m), val=amp_series[f"{wave_name}_amp"]))
    
    return partial_wave_amplitudes

def parse_amplitude_vecps(amp: str) -> Tuple[int, int, int, int, int]:
    """ interface Kevin's expected notation to PyAmpTools notation"""
    e, L, M, J = converter[amp] # [int, int, int, int]
    P = (-1) ** L
    return (e, J, P, M, L)
