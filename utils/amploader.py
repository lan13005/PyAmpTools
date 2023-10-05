# import os
import numpy as np

############################################################
# This file holds scripts to load in amptools information
############################################################

class AmplitudeParameters:
    '''
    Class to extract amplitude parameters from a FitResults or ConfigurationInfo object
       Ability to format them (complex -> real, imag) for input to other minimization algorithms
    '''
    def __init__(self):
        self.uniqueAmps = {}
        self.uniqueReal = {}

    def load_cfg(self, cfg): # cfg: [FitResults, ConfigurationInfo]
        '''
        Get a map of unique (amplitude: value) pairs excluding extra constrained ones.
        For a FitResults object set values to fitted results
        For a ConfigurationInfo object set values to initial values
        '''
        ######## GET UNIQUE AMPLITUDES ########
        if hasattr(cfg, 'configInfo'): # input was a FitResults object
            results, cfg = cfg, cfg.configInfo()
            ftype = "FitResults"
        elif hasattr(cfg, 'constraintMap'):
            ftype = "ConfigurationInfo" # input was a ConfigurationInfo object
        else:
            raise ValueError("Input must be a FitResults or ConfigurationInfo object")
        constraintMap = cfg.constraintMap()
        constraint_index = {}
        uniqueAmps = []
        i = 0
        for k, vs in constraintMap:
            if k not in constraint_index and not cfg.amplitude(k).fixed():
                constraint_index[k] = i
                uniqueAmps.append(k)
            for v in vs:
                if v not in constraint_index and not cfg.amplitude(v).fixed():
                    constraint_index[v] = i
            i += 1
        ######## GET VALUES / REALNESS OF UNIQUE AMPLITUDES ########
        self.uniqueAmps = {k: (results.ampProdParMap()[k] if ftype=="FitResults" else cfg.amplitude(k)) for k in uniqueAmps}
        self.uniqueReal = {k: cfg.amplitude(k).real() for k in uniqueAmps} # check if amplitude is set to be real

    def flatten_parameters(self, uniqueAmps={}, uniqueReal={}):
        ''' Flatten amplitude parameters (complex-> real, imag) skipping imaginary parts of real amplitudes.
         Dictionary to Array
        '''
        parameters = []
        if len(uniqueAmps) == 0:
            uniqueAmps = self.uniqueAmps
            uniqueReal = self.uniqueReal
        for k, v in uniqueAmps.items():
            parameters.append(v.real)
            if not uniqueReal[k]:
                parameters.append(v.imag)
        return parameters

    def unflatten_parameters(self, parameters, uniqueReal={}):
        ''' Unflatten amplitude parameters forming complex values again.
         Array to Dictionary
        '''
        uniqueAmps = {}
        parameters = parameters.copy() # don't pop original list
        if len(uniqueReal) == 0:
            uniqueReal = self.uniqueReal
        i=0
        for k, real in uniqueReal.items():
            real_part = parameters[i]; i += 1
            imag_part = 0
            if not real:
                imag_part = parameters[i]; i+=1
            uniqueAmps[k] = complex(real_part, imag_part)
        return uniqueAmps

    def get_naming_convention(self):
        '''
        amp_names: array of amplitude names.
        amp_names_parts: array of amplitude names with real/imag parts separated, matches flattened_parameters
        par_indices: array of ints corresponding to real/imag parts of amplitudes, reals are even, imags are odd
        '''
        amp_names = list(self.uniqueAmps.keys())
        par_indices, amp_names_parts = [], []
        for i, (amp, real) in enumerate(self.uniqueReal.items()):
            par_indices.append(2*i)
            amp_names_parts.append(f'{amp}_re')
            if not real:
                par_indices.append(2*i+1)
                amp_names_parts.append(f'{amp}_im')
        return np.array(amp_names), np.array(amp_names_parts), np.array(par_indices)
