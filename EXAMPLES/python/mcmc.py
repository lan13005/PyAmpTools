import numpy as np
import emcee
import os
from typing import List
import matplotlib.pyplot as plt
from LoadParameters import LoadParameters, createMovesMixtureFromDict
import corner
import argparse
import time
import sys
import yaml
import atiSetup
from utils import glob_sort_captured, safe_getsize

class mcmcManager:
    '''
    This class manages the MCMC sampling process and the interaction with AmpToolsInterface instances
    '''
    def __init__(self, atis, LoadParametersSamplers, ofile):
        self.atis = atis
        self.LoadParametersSamplers = LoadParametersSamplers
        self.ofile = ofile
        self.finishedSetup = False

    def Prior(self, parameters):
        '''
        (Log of) Prior distribution for the parameters

        Args:
            parameters (dict): Dictionary of parameter names (complex production coeffs and amplitude params) and values.
                If multiple config files were passed, the parameter names will be appended with a '_i' tag where i is the ith cfg file

        Returns:
            prior (float): Log prior probability
        '''
        return 0

    def Likelihood(self, ati):
        ''' Returns Log likelihood from AmpToolsInterface instance. Separation for profiling '''
        return -ati.likelihood()

    def LogProb(
        self,
        par_values,
        keys,
        ):
        '''
        Definition of the (Log) Posterior distribution

        Args:
            par_values (float): Flattened (complex-to-Real/Imag) Parameter values. ~ [real1, imag1, real2, imag2, ...]
            keys (str): Flattened Parameter names. ~ [amp1,  amp1,  amp2,  amp2, ...]
                If multiple config files were passed, the parameter names will be appended with a '_i' tag where i is the ith cfg file

        Returns:
            log_prob (float): Log posterior probability
        '''

        ## Calculate Log likelihood
        log_prob = 0
        n_atis = len( self.atis )
        for i, ati, LoadParametersSampler in zip( range(n_atis), self.atis, self.LoadParametersSamplers ):
            _par_values = par_values[self.paramPartitions[i]:self.paramPartitions[i+1]]
            _keys = keys[self.paramPartitions[i]:self.paramPartitions[i+1]]
            parameters = LoadParametersSampler.unflatten_parameters( _par_values, _keys )
            for name, value in parameters.items():
                # remove tag from name if multiple cfg files were passed
                name = name[ :name.rfind('_') ] if n_atis > 1 else name
                ati.parameterManager()[name] = value

            log_prob += self.Likelihood(ati) + self.Prior(parameters)

        return log_prob

    def perform_mcmc(
            self,
            burnIn:    int = 100,
            nwalkers:  int = 32,
            nsamples:  int = 1000,
            ### Non-CLI arguments Below ###
            params_dict: dict = {},
            moves_mixture = [ emcee.moves.StretchMove() ],
            sampler_kwargs = {},
        ):

        '''
        Performs MCMC sampling

        Args:
            burnIn (int): number of burn-in steps
            nwalkers (int): number of walkers
            nsamples (int): number of samples
            params_dict (dict): Dictionary of parameter names (complex production coeffs and amplitude params) and values.
            moves_mixture [ (emcee.moves.{move}(kwargs), probability) ]: List of tuples of moves (with kwargs pluggin in) and their probability
            sampler_kwargs (dict): Additional keyword arguments to pass to emcee.EnsembleSampler()

        Returns:
            results (dict): Dictionary containing results of MCMC sampling (samples, MAP, acceptance fraction, autocorrelation time, etc.)
        '''

        print(f'\n ================== PERFORM_MCMC() KWARGS CONFIGURATION ================== ')
        print(f'nwalkers: {nwalkers}')
        print(f'burnIn: {burnIn}')
        print(f'nsamples: {nsamples}')
        print(f'params_dict: {params_dict}')
        print(f'moves_mixture: {moves_mixture}')
        print(f'sampler_kwargs: {sampler_kwargs}')
        print(f' =================================================================== \n')

        ########## SETUP PARAMETERS ############
        ## AmpTools config files are assumed to contain the MLE values already
        ##   this can be acheived by running fit.py with --seedfile
        ##   and using the include directive in the config file to add the seedfile
        ##   to override values
        ## doNotScatter = True  will initialize walkers in an N-ball around the MLE estimate
        ## doNotScatter = False will uniformly sample (or scatter) on user-specified interval
        params = {}
        doNotScatter = True
        if len(params_dict) != 0:
            first_element = next(iter(params_dict.values()))
            doNotScatter = type(first_element) != list or len(params_dict)==0
            params = params_dict if doNotScatter else {k:v[0] for k,v in params_dict.items()}

        ####### LOAD PARAMETERS #####
        mle_values, keys, par_names_flat = [], [], []
        paramPartitions = [0]
        for i, LoadParametersSampler in enumerate(self.LoadParametersSamplers):
            mles, ks, pars = LoadParametersSampler.flatten_parameters(params)
            mle_values.extend(mles)
            tag = f'_{i}' if len(self.LoadParametersSamplers) > 1 else ''
            keys.extend( [ f'{k}{tag}' for k in ks ] )
            par_names_flat.extend( [ f'{par}{tag}' for par in pars ] )
            paramPartitions.append( len(mles) + paramPartitions[-1] )
        self.paramPartitions = paramPartitions

        ####### ADDITIONAL BASIC SETUP ########
        if nwalkers < 2*len(mle_values):
            print(f"\n[emcee req.] Number of walkers must be at least twice the number of \
                  parameters. Overwriting nwalkers to 2*len(mle_values) = 2*{len(mle_values)}\n")
            nwalkers = 2*len(mle_values)

        nDim = len(mle_values)

        ############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
        fit_start_time = time.time()
        output_file = f'{self.ofile}'
        fileTooSmall = safe_getsize(output_file) < 1e4 # remake if too small, likely corrupted initialization
        if not os.path.exists( output_file ) or fileTooSmall:
            if fileTooSmall:
                os.system(f'rm -f {output_file}')

            print(f' ================== RUNNING MCMC ================== ')
            if doNotScatter:
                print("Initializing walkers in an N-ball around the MLE estimate")
                par_values = np.array(mle_values)
                par_values = np.repeat(par_values, nwalkers).reshape(nDim, nwalkers).T
                par_values[par_values==0] += 1e-2 # avoid 0 values, leads to large condition number for sampler
                par_values *= ( 1 + 0.01 * np.random.normal(0, 1, size=(nwalkers, nDim)) )
            else:
                print("Randomly sampling on user-specified interval")
                par_values = np.empty((nwalkers, nDim))
                for k, (_, mini, maxi) in params_dict.items():
                    par_values[:, keys.index(k)] = np.random.uniform(mini, maxi, size=nwalkers)

            backend = emcee.backends.HDFBackend(output_file)
            backend.reset(nwalkers, nDim)
            sampler = emcee.EnsembleSampler(nwalkers, nDim, self.LogProb,
                                            args=[keys],
                                            moves=moves_mixture,
                                            backend=backend)

            _sampler_kwargs = { 'progress': True }
            _sampler_kwargs.update(sampler_kwargs)
            if burnIn != 0:
                print(f'\n[Burn-in beginning]\n')
                state = sampler.run_mcmc(par_values, burnIn)
                sampler.reset()
                print(f'\n[Burn-in complete. Running sampler]\n')
                sampler.run_mcmc(state, nsamples, **_sampler_kwargs);
            else:
                sampler.run_mcmc(par_values, nsamples, **_sampler_kwargs);

            print(f'\n[Sampler complete]\n')

        else: # Found already existing HDF5 file, will just load it
            print(f' ================== LOADING MCMC ================== ')
            sampler = emcee.backends.HDFBackend(output_file)
        fit_end_time = time.time()

        ################# CALCULATE AUTOCORR AND ACCEPTANCE FRACTION #################
        samples = sampler.get_chain(flat=True) # returns (nsamples*nwalkers, 5) flattened array
        if isinstance(sampler, emcee.EnsembleSampler):
            acceptance_fraction = np.mean(sampler.acceptance_fraction)
        else: # HDF5 backend
            # function implementation in github, acceptance_fraction not available from HDF5 backend
            acceptance_fraction = np.mean(sampler.accepted / sampler.iteration)
        autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
        print(f"Autocorrelation time: {autocorr_time:.3f} steps")

        ################### COMPUTE MAP ESTIMATE ###################
        map_idx = np.argmax(sampler.get_log_prob(flat=True)) # maximum a posteriori (MAP) location
        map_estimate = samples[map_idx,:]

        print(' =================== RESULTS =================== ')
        print(f'MAP Estimates from {len(samples)} samples obtained over {nwalkers} walkers:')
        [print(f'   {l:20} = {v:0.3f}') for l, v in zip(par_names_flat, map_estimate)]
        print(' =============================================== ')

        ################### TRACK SOME ATTRIBUTES ###################
        self.nwalkers = nwalkers
        self.samples = samples
        self.mle_values = mle_values
        self.map_estimate = map_estimate
        self.paramPartitions = paramPartitions # list of indices to re-group par_names_flat into each cfg file
        self.par_names_flat = par_names_flat
        self.elapsed_fit_time = fit_end_time - fit_start_time

    def draw_corner(
        self,
        corner_ofile = '',
        save = True,
        kwargs = {},
    ):
        '''
        Draws a corner plot to visualize sampling and parameter correlations

        Args:
            corner_ofile (str): Output file name
            save (bool): Save the figure to a file or return figure. Default True = Save
            safe_draw (bool):
            kwargs (dict): Keyword arguments to pass to corner.corner()

        Returns:
            fig (matplotlib.figure.Figure): Figure object if save=False
        '''

        if corner_ofile == '':
            print("No corner plot output file specified. Not drawing corner plot")
            return

        def plot_axvh_fit_params( # Yes, some function inception going on
                values,  # [real1, imag1, real2, imag2, ...]
                nDim,    # number of parameters
                color='black'
            ):
            ''' Draws lines on the corner plot to indicate the specific parameter values '''
            for yi in range(nDim):
                for xi in range(yi):
                    axes[yi, xi].axvline(values[xi], color=color, linewidth=4, alpha=0.8)
                    axes[yi, xi].axhline(values[yi], color=color, linewidth=4, alpha=0.8)
                axes[yi, yi].axvline(values[yi], color=color, linewidth=4, alpha=0.8)

        ####### DRAW CORNER PLOT #######
        corner_kwargs = {"color": "black", "show_titles": True }
        corner_kwargs.update(kwargs)
        labels_ReIm = [f'{l.split("::")[-1]}' for l in self.par_names_flat]
        nDim = self.samples.shape[1]
        fig = corner.corner( self.samples, labels=labels_ReIm, **corner_kwargs )
        axes = np.array(fig.axes).reshape((nDim, nDim))
        plot_axvh_fit_params(self.map_estimate, nDim, color='royalblue')
        plot_axvh_fit_params(self.mle_values, nDim, color='tab:green')

        if save: plt.savefig(f"{corner_ofile}")
        else:    return fig

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='emcee fitter')
    parser.add_argument('--cfgfiles', type=str, default='', help='Config file name or glob pattern')
    parser.add_argument('--nwalkers', type=int, default=32, help='Number of walkers. Default 32')
    parser.add_argument('--burnin', type=int, default=100, help='Number of burn-in steps. Default 100')
    parser.add_argument('--nsamples', type=int, default=1000, help='Number of samples. Default 1000')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if exists')
    parser.add_argument('--accelerator', type=str, default='mpigpu', help='Use accelerator if available ~ [cpu, gpu, mpi, mpigpu, gpumpi]')
    parser.add_argument('--ofile', type=str, default='mcmc/mcmc.h5', help='Output file name. Default "mcmc/mcmc.h5"')
    parser.add_argument('--corner_ofile', type=str, default='', help='Corner plot output file name. Default empty str = do not draw')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for consistent walker random initialization. Default 42')
    parser.add_argument('--yaml_override', type=str, default={}, help='Path to YAML file containing parameter overrides. Default "" = no override')

    args = parser.parse_args(sys.argv[1:])

    yaml_override = {}
    if args.yaml_override != {}:
        with open(args.yaml_override, 'r') as f:
            yaml_override = yaml.safe_load(f)

    ################ Override CLI arguments with YAML file if specified ###################
    cfgfiles  = args.cfgfiles if 'cfgfiles' not in yaml_override else yaml_override['cfgfiles']
    ofile    = args.ofile if 'ofile' not in yaml_override else yaml_override['ofile']
    corner_ofile = args.corner_ofile if 'corner_ofile' not in yaml_override else yaml_override['corner_ofile']
    nwalkers = args.nwalkers if 'nwalkers' not in yaml_override else yaml_override['nwalkers']
    burnIn   = args.burnin if 'burnin' not in yaml_override else yaml_override['burnin']
    nsamples = args.nsamples if 'nsamples' not in yaml_override else yaml_override['nsamples']
    overwrite_ofile = args.overwrite if 'overwrite' not in yaml_override else yaml_override['overwrite']
    seed     = args.seed if 'seed' not in yaml_override else yaml_override['seed']
    params_dict = {} if 'params_dict' not in yaml_override else yaml_override['params_dict']
    moves_mixture = [ emcee.moves.StretchMove() ] if 'moves_mixture' not in yaml_override else \
                        createMovesMixtureFromDict(yaml_override['moves_mixture'])

    assert( cfgfiles != '' ), 'You must specify a config file'

    cfgfiles = glob_sort_captured(cfgfiles) # list of sorted config files based on captured number
    for cfgfile in cfgfiles:
        assert( os.path.exists(cfgfile) ), 'Config file does not exist at specified path'

    ################### LOAD LIBRARIES ##################
    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE #############
    parsers  = [ ConfigFileParser(cfgfile) for cfgfile in cfgfiles ]
    cfgInfos = [ parser.getConfigurationInfo() for parser in parsers ] # List of ConfigurationInfo

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerAmplitude( Vec_ps_refl() )
    AmpToolsInterface.registerAmplitude( OmegaDalitz() )
    AmpToolsInterface.registerAmplitude( BreitWigner() )
    AmpToolsInterface.registerAmplitude( Piecewise() )
    AmpToolsInterface.registerAmplitude( PhaseOffset() )
    AmpToolsInterface.registerAmplitude( TwoPiAngles() )
    AmpToolsInterface.registerDataReader( DataReader() )
    AmpToolsInterface.registerDataReader( DataReaderTEM() )
    AmpToolsInterface.registerDataReader( DataReaderFilter() )

    atis = [ AmpToolsInterface(cfgInfo) for cfgInfo in cfgInfos ]
    LoadParametersSamplers = [ LoadParameters(cfgInfo) for cfgInfo in cfgInfos ]

    ############## RUN MCMC ##############
    np.random.seed(seed)

    if RANK_MPI == 0:
        print("\n ====================================================================================")
        print(f" cfgfiles: {cfgfiles}")
        print(f" ofile: {ofile}")
        print(f" corner_ofile: {corner_ofile}")
        print(f" nwalkers: {nwalkers}")
        print(f" burnIn: {burnIn}")
        print(f" nsamples: {nsamples}")
        print(f" overwrite_ofile: {overwrite_ofile}")
        print(f" seed: {seed}")
        print(f" params_dict: {params_dict}")
        print(f" moves_mixture: {moves_mixture}")
        print(" ====================================================================================\n")

        print(f'\n===============================')
        print(f'Loaded {len(cfgInfos)} config files')
        print(f'===============================\n')
        displayN = 1 # Modify me to show display more cfg files. Very verbose!
        for cfgfile, cfgInfo in zip(cfgfiles[:displayN], cfgInfos[:displayN]):
            print('\n=============================================')
            print(f'Display Of This Config File Below: {cfgfile}')
            print('=============================================\n')
            cfgInfo.display()

        ############## PREPARE FOR SAMPLER ##############
        if os.path.isfile(f'{ofile}') and overwrite_ofile:
            os.system(f'rm -f {ofile}')
            print("Overwriting existing output file!")
        if '/' in ofile:
            ofolder = ofile[:ofile.rfind("/")]
            os.system(f'mkdir -p {ofolder}')

        mcmcMgr = mcmcManager(atis, LoadParametersSamplers, ofile)

        mcmcMgr.perform_mcmc(
            nwalkers = nwalkers,
            burnIn   = burnIn,
            nsamples = nsamples,
            params_dict   = params_dict,
            moves_mixture = moves_mixture,
        )

        mcmcMgr.draw_corner(f'{corner_ofile}')

        print(f"Fit time: {mcmcMgr.elapsed_fit_time} seconds")
        print(f"Total time: {time.time() - start_time} seconds")
