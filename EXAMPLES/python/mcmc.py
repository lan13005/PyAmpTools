import numpy as np
import emcee
import os
from typing import List
import matplotlib.pyplot as plt
from LoadParameters import LoadParameters
import corner
import argparse
import time
import sys
import yaml
import atiSetup

def LogProb(
    par_values,
    keys,
    ati,
    LoadParametersSampler
    ):
    '''
    Definition of the (Log) Posterior distribution

    Args:
        par_values (float): Flattened (complex-to-Real/Imag) Parameter values. ~ [real1, imag1, real2, imag2, ...]
        keys (str): Flattened Parameter names. ~ [amp1,  amp1,  amp2,  amp2, ...]
        ati (AmpToolsInterface): AmpToolsInterface instance
        LoadParametersSampler (LoadParameters): LoadParameters class to manage parameters

    Returns:
        log_prob (float): Log posterior probability
    '''

    ## Calculate Log likelihood
    ll = -1e7
    parameters = LoadParametersSampler.unflatten_parameters(par_values, keys)
    for name, value in parameters.items():
        # parameterManager has been pythonized to act like a dictionary
        ati.parameterManager()[name] = value
    ll = -ati.likelihood()

    ## Add lasso prior on parameter values we know are small
    # lasso_strength = 0.1
    prior = 0 # -lasso_strength * np.sum(np.abs(par_values))

    log_prob = ll + prior
    # print(f'LogProb: {log_prob} = {ll} + {prior}')
    return log_prob

def createMovesMixtureFromDict(moves_dict):
    '''
    Creates a mixture of moves for the emcee sampler

    Args:
        moves_dict { move: {kwargs: {}, probability} }: Dictionary of moves and their kwargs and probability

    Returns:
        moves_mixture [ (emcee.moves.{move}(kwargs), probability) ]: List of tuples of moves (with kwargs pluggin in) and their probability
    '''
    moves_mixture = []
    for move, moveDict in moves_dict.items():
        move = eval(f'emcee.moves.{move}') # convert string to class
        kwargs = moveDict['kwargs']
        prob = moveDict['prob']
        moves_mixture.append( (move(**kwargs), prob) )
    return moves_mixture

def perform_mcmc(
        ati,
        LoadParametersSampler,
        ofolder:   str = 'mcmc',
        ofile:     str = 'mcmc.h5',
        nwalkers:  int = 32,
        burnIn:    int = 100,
        nsamples:  int = 1000,
        ### Non-CLI arguments Below ###
        params_dict   = {},
        moves_mixture = [ emcee.moves.StretchMove() ],
        sampler_kwargs = {},
    ):

    '''
    Performs MCMC sampling

    Args:
        ati (AmpToolsInterface): AmpToolsInterface instance
        LoadParametersSampler (LoadParameters): LoadParameters instance
        ofolder (str): output folder name
        ofile (str): output file name
        nwalkers (int): number of walkers
        burnIn (int): number of burn-in steps
        nsamples (int): number of samples
        params_dict (dict): {par: value} or {par: [value, min, max]} where sample from [min, max] for walker initialization
        moves_mixture [ (emcee.moves.{move}(kwargs), probability) ]: List of tuples of moves (with kwargs pluggin in) and their probability
        sampler_kwargs (dict): Additional keyword arguments to pass to emcee.EnsembleSampler()

    Returns:
        results (dict): Dictionary containing results of MCMC sampling (samples, MAP, acceptance fraction, autocorrelation time, etc.)
    '''

    print(f'\n ================== PERFORM_MCMC() KWARGS CONFIGURATION ================== ')
    print(f'ofolder: {ofolder}')
    print(f'ofile: {ofile}')
    print(f'nwalkers: {nwalkers}')
    print(f'burnIn: {burnIn}')
    print(f'nsamples: {nsamples}')
    print(f'params_dict: {params_dict}')
    print(f'moves_mixture: {moves_mixture}')
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
    mle_values, keys, par_names_flat  = LoadParametersSampler.flatten_parameters(params)

    if nwalkers < 2*len(mle_values):
        print(f"\n[emcee req.] Number of walkers must be at least twice the number of \
              parameters. Overwriting nwalkers to 2*len(mle_values) = 2*{len(mle_values)}\n")
        nwalkers = 2*len(mle_values)

    nDim = len(mle_values)

    ############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
    fit_start_time = time.time()
    if not os.path.exists(f'{ofolder}/{ofile}'):

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

        backend = emcee.backends.HDFBackend(f'{ofolder}/{ofile}')
        backend.reset(nwalkers, nDim)
        sampler = emcee.EnsembleSampler(nwalkers, nDim, LogProb,
                                        args=[keys, ati, LoadParametersSampler],
                                        moves=moves_mixture,
                                        backend=backend)

        print(f'Par values: {par_values}')

        _sampler_kwargs = {'progress': True}
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
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
        print(f"\nMean acceptance fraction: {acceptance_fraction:.3f}")
        print(f"Autocorrelation time: {autocorr_time:.3f} steps\n")

    else:
        print(f' ================== LOADING MCMC ================== ')
        sampler = emcee.backends.HDFBackend(f'{ofolder}/{ofile}')
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

    ####################
    ## MASK OUT BAD SAMPLES. This is likely a sign of something bad but lets ignore it for now
    ####################
    MIN_VAL = -1e10
    MAX_VAL =  1e10
    mask = np.logical_and(samples > MIN_VAL, samples < MAX_VAL).all(axis=1)
    nsamples_before_mask = samples.shape[0]
    samples = samples[mask]
    print(f"Percent samples remaning after masking: {100*samples.shape[0]/nsamples_before_mask:0.2f}%")

    ################### COMPUTE MAP ESTIMATE ###################
    map_idx = np.argmax(sampler.get_log_prob(flat=True)) # maximum a posteriori (MAP) location
    map_estimate = samples[map_idx,:]

    return {'samples': samples, 'map_estimate': map_estimate, 'autocorr_time': autocorr_time,
            'acceptance_fraction': acceptance_fraction, 'mle_values': mle_values, 'par_names_flat': par_names_flat,
            'nDim': nDim, 'nwalkers': nwalkers, 'nsamples': nsamples, 'elapsed_fit_time': fit_end_time - fit_start_time,}

################### PLOT RESULTS ###################
def draw_corner(
    results,
    corner_ofile_path = 'corner.png',
    save = True,
    kwargs = {},
):
    '''
    Draws a corner plot to visualize sampling and parameter correlations

    Args:
        results (dict): Dictionary containing results of MCMC sampling (samples, MAP, acceptance fraction, autocorrelation time, etc.)
        corner_ofile_path (str): output file path
        save (bool): Save the figure to a file or return figure. Default True = Save
        kwargs (dict): Keyword arguments to pass to corner.corner()

    Returns:
        fig (matplotlib.figure.Figure): Figure object if save=False
    '''

    def plot_axvh_fit_params( # Yes, some function inception going on
            values,  # [real1, imag1, real2, imag2, ...]
            color='black'
        ):
        ''' Draws lines on the corner plot to indicate the specific parameter values '''
        for yi in range(nDim):
            for xi in range(yi):
                axes[yi, xi].axvline(values[xi], color=color, linewidth=4, alpha=0.8)
                axes[yi, xi].axhline(values[yi], color=color, linewidth=4, alpha=0.8)
            axes[yi, yi].axvline(values[yi], color=color, linewidth=4, alpha=0.8)

    ####### LOAD REQUIRED VARIABLES FROM RESULTS DICTIONARY #######
    samples        = results['samples']
    map_estimate   = results['map_estimate']
    mle_values     = results['mle_values']
    par_names_flat = results['par_names_flat']
    nwalkers       = results['nwalkers']

    ####### DRAW CORNER PLOT #######
    corner_kwargs = {"color": "black", "show_titles": True }
    corner_kwargs.update(kwargs)
    print(f'Corner kwargs: {corner_kwargs}')
    labels_ReIm = [f'{l.split("::")[-1]}' for l in par_names_flat]
    fig = corner.corner(samples, labels=labels_ReIm, **corner_kwargs)
    nDim = samples.shape[1]
    axes = np.array(fig.axes).reshape((nDim, nDim))
    plot_axvh_fit_params(map_estimate, color='royalblue')
    plot_axvh_fit_params(mle_values, color='tab:green')

    print(' =================== RESULTS =================== ')
    print(f'MAP Estimates from {len(samples)} samples obtained over {nwalkers} walkers:')
    [print(f'   {l:20} = {v:0.3f}') for l, v in zip(par_names_flat, map_estimate)]
    print(' =============================================== ')

    # import pygtc
    # GTC_ReIM = pygtc.plotGTC(chains=[samples], paramNames=par_names_flat,
    #                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page',
    #                     plotName=f'{ofolder}/mcmc_ReIm.pdf', nContourLevels=3)
    # GTC = pygtc.plotGTC(chains=[intensity], paramNames=PAR_NAMES,
    #                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page',
    #                     plotName=f'{ofolder}/mcmc.pdf', nContourLevels=3)

    if save:
        plt.savefig(f"{corner_ofile_path}")
    else:
        return fig

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='emcee fitter')
    parser.add_argument('--cfgfile', type=str, default='', help='Config file name')
    parser.add_argument('--nwalkers', type=int, default=32, help='Number of walkers. Default 32')
    parser.add_argument('--burnin', type=int, default=100, help='Number of burn-in steps. Default 100')
    parser.add_argument('--nsamples', type=int, default=1000, help='Number of samples. Default 1000')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if exists')
    parser.add_argument('--accelerator', type=str, default='mpigpu', help='Use accelerator if available ~ [cpu, gpu, mpi, mpigpu, gpumpi]')
    parser.add_argument('--ofolder', type=str, default='mcmc', help='Output folder name. Default "mcmc"')
    parser.add_argument('--ofile', type=str, default='mcmc.h5', help='Output file name. Default "mcmc.h5"')
    parser.add_argument('--corner_ofile', type=str, default='corner.png', help='Corner plot output file name. Default "corner.png"')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for consistent walker random initialization. Default 42')
    parser.add_argument('--yaml_override', type=str, default={}, help='Path to YAML file containing parameter overrides. Default "" = no override')

    args = parser.parse_args(sys.argv[1:])

    yaml_override = {}
    if args.yaml_override != {}:
        with open(args.yaml_override, 'r') as f:
            yaml_override = yaml.safe_load(f)

    cfgfile  = args.cfgfile if 'cfgfile' not in yaml_override else yaml_override['cfgfile']
    ofolder  = args.ofolder if 'ofolder' not in yaml_override else yaml_override['ofolder']
    ofile    = args.ofile if 'ofile' not in yaml_override else yaml_override['ofile']
    corner_ofile = args.corner_ofile if 'corner_ofile' not in yaml_override else yaml_override['corner_ofile']
    nwalkers = args.nwalkers if 'nwalkers' not in yaml_override else yaml_override['nwalkers']
    burnIn   = args.burnin if 'burnin' not in yaml_override else yaml_override['burnin']
    nsamples = args.nsamples if 'nsamples' not in yaml_override else yaml_override['nsamples']
    overwrite_ofile = args.overwrite if 'overwrite' not in yaml_override else yaml_override['overwrite']
    seed     = args.seed if 'seed' not in yaml_override else yaml_override['seed']
    params_dict = {} if 'params_dict' not in yaml_override else yaml_override['params_dict']
    moves_mixture = [ emcee.moves.StretchMove() ] if 'moves_mixture' not in yaml_override else createMovesMixtureFromDict(yaml_override['moves_mixture'])

    assert( cfgfile != '' ), 'You must specify a config file'

    print("\n ====================================================================================")
    print(f" cfgfile: {cfgfile}")
    print(f" ofolder: {ofolder}")
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

    ############## PREPARE FOR SAMPLER ##############
    assert( os.path.exists(cfgfile) ), 'Config file does not exist at specified path'
    if os.path.isfile(f'{ofolder}/{ofile}') and overwrite_ofile:
        os.system(f'rm {ofolder}/{ofile}')
        print("Overwriting existing output file!")
    os.system(f'mkdir -p {ofolder}')

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE ##############
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
    cfgInfo.display()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerAmplitude( BreitWigner() )
    AmpToolsInterface.registerAmplitude( Piecewise() )
    AmpToolsInterface.registerAmplitude( PhaseOffset() )
    AmpToolsInterface.registerAmplitude( TwoPiAngles() )
    AmpToolsInterface.registerDataReader( DataReader() )
    AmpToolsInterface.registerDataReader( DataReaderFilter() )

    ati = AmpToolsInterface( cfgInfo )
    LoadParametersSampler = LoadParameters()
    LoadParametersSampler.load_cfg( cfgInfo )

    ############## RUN MCMC ##############
    np.random.seed(seed)

    results = perform_mcmc(ati,
                           LoadParametersSampler,
                           ofolder   = ofolder,
                           ofile     = ofile,
                           nwalkers  = nwalkers,
                           burnIn    = burnIn,
                           nsamples  = nsamples,
                           params_dict   = params_dict,
                           moves_mixture = moves_mixture,
                           )
    elapsed_fit_time = results['elapsed_fit_time']
    draw_corner(results, f'{ofolder}/{corner_ofile}')

    print(f"Fit time: {elapsed_fit_time} seconds")
    print(f"Total time: {time.time() - start_time} seconds")
