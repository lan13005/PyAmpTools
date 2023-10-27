# import os
import numpy as np

############################################################
# This file holds scripts to load in amptools information
############################################################

class LoadParameters:
    '''
    Class to extract amplitude parameters from an AmpTools FitResults or ConfigurationInfo object
       Ability to format them (complex -> real, imag) for input to other minimization algorithms
    '''
    def __init__(self):
        self.uniqueProdPars = {}
        self.uniqueProdIsReal = {}

    def load_cfg(self, cfg): # cfg: [FitResults, ConfigurationInfo]
        '''
        Get a map of unique (parameter: value) pairs excluding additional constrained ones
            These include production parameters and amplitude parameters
        
        Args:
            cfg: If cfg is a FitResults object set values to fitted results. For a ConfigurationInfo object set values to initial values
        '''
        ######## GET UNIQUE AMPLITUDES' PRODUCTION PARAMETERS ########
        results = None
        if hasattr(cfg, 'configInfo'): # input was a FitResults object
            print("Input was a FitResults object. Take actions accordingly...")
            results, cfg = cfg, cfg.configInfo()
            ftype = "FitResults"
        elif hasattr(cfg, 'constraintMap'):
            print("Input was a ConfigurationInfo object. Take actions accordingly...")
            ftype = "ConfigurationInfo" # input was a ConfigurationInfo object
        else:
            raise ValueError("Input must be a FitResults or ConfigurationInfo object")
        self.results = results
        self.cfg     = cfg

        constraintMap = cfg.constraintMap()
        constraint_index = {}
        uniqueProdPars   = []
        i = 0
        for k, vs in constraintMap:
            if k not in constraint_index and not cfg.amplitude(k).fixed():
                constraint_index[k] = i
                uniqueProdPars.append(k)
            for v in vs:
                if v not in constraint_index and not cfg.amplitude(v).fixed():
                    constraint_index[v] = i
            i += 1

        ######## GET VALUES / REALNESS OF UNIQUE AMPLITUDES' PRODUCTION PARAMETERS ########
        self.uniqueProdPars   = {k: (results.ampProdParMap()[k] if ftype=="FitResults" else cfg.amplitude(k).value()) for k in uniqueProdPars}
        self.uniqueProdIsReal = {k: cfg.amplitude(k).real() for k in uniqueProdPars} # check if amplitude is set to be real

        ####### GET AMPLITUDE PARAMETERS ########
        # parameters associated with amplitudes (i.e. masses, widths, etc)
        ampPars = cfg.parameterList() # ParameterInfo*
        self.ampPars = {}
        for par in ampPars:
            if not par.fixed():
                self.ampPars[par.parName()] = results.ampParMap()[par.parName()] if ftype=="FitResults" else par.value()
        self.nAmpPars = len(ampPars)

        ####### MERGE DICTIONARIES ########
        self.params       = self.uniqueProdPars   | self.ampPars # python 3.9 - merge dictionaries
        self.paramsIsReal = self.uniqueProdIsReal | {k: True for k in self.ampPars.keys()} # all amp params are real

    def flatten_parameters(self, params={}):
        '''
        Flatten amplitude parameters (complex-> real, imag) skipping imaginary parts of real amplitudes.
        If no arguments are passed, use the uniqueProdPars and uniqueProdIsReal from the last call to load_cfg()
        Can also format any dictionary pair into flat format
            Dictionary to List

        Args:
            params (dict): dictionary of parameters to flatten

        Returns:
            parameters (list): list of flattened parameters
            key (list): list of keys (parameter names) corresponding to parameters
            names (list): list of parameter names expanding complex parameters into Re[par] and Im[par]
        '''
        parameters = []
        key        = [] # lose information on flatten, use key to keep track of provenance
        names      = [] # track parameter names, appending re/im to name if complex
        if len(params)==0:
            params = self.params
        for k, v in params.items():
            assert( k in self.paramsIsReal.keys() ), f'Parameter {k} not found in parameter list! Is parameter fixed?'
            pk = k.split('::')[-1] # Production amplitudes take form "Reaction::Sum::Amp" grab Amp, pretty_k = pk
            real_value = v.real      if k in self.uniqueProdPars else v
            names.append(f'Re[{pk}]' if k in self.uniqueProdPars else pk)
            parameters.append( real_value )
            key.append(k)
            if not self.paramsIsReal[k]:
                parameters.append(v.imag)
                key.append(k)
                names.append(f'Im[{pk}]')
        return parameters, key, names

    def unflatten_parameters(self, parameters, key):
        '''
        Unflatten parameters forming complex values (for production parameters) when requested.
            List to Dictionary

        Args:
            parameters (list): list of flattened parameters
            key (list): list of keys (parameter names) corresponding to parameters

        Returns:
            paramDict (dict): dictionary of parameters

        '''
        paramDict = {}
        for k, param in zip(key,parameters):
            if k not in paramDict: paramDict[k] = param
            else: paramDict[k] += 1j*param # a repeat in key means imaginary part
        return paramDict
