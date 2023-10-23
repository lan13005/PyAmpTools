import numpy as np
import numpy.typing as npt
import emcee
import ROOT
import os
from typing import List
import matplotlib.pyplot as plt
from LoadParameters import LoadParameters
import corner
import argparse
import time
import sys
from utils import prepare_mpigpu

def LogProb(
    par_values,           # [real1, imag1, real2, imag2, ...]
    keys,                 # [amp1,  amp1,  amp2,  amp2, ...]
    ati,                  # AmpToolsInterface
    LoadParametersSampler # LoadParameters class
    ):
    '''
    Posterior distribution definition
        Log probability function = Log Likelihood if no prior
    '''

    ## Calculate Log likelihood
    ll = -1e7
    parameters = LoadParametersSampler.unflatten_parameters(par_values, keys)
    for name, value in parameters.items():
        # parameterManager has been pythonized to act like a dictionary
        ati.parameterManager()[name] = value
    ll = -ati.likelihood()

    ## Add lasso prior on parameter values we know are small
    # lasso = 0.1
    prior = 0 # -lasso * np.sum(np.abs(par_values[PAR_INDICES >= 2]))

    log_prob = ll + prior
    # print(f'LogProb: {log_prob} = {ll} + {prior}')
    return log_prob


def perform_mcmc(
        ati,                        # AmpToolsInterface class
        LoadParametersSampler,      # LoadParameters class
        ofolder:   str = 'mcmc',    # output folder name
        ofile:     str = 'mcmc.h5', # output file name
        nwalkers:  int = 32,        # number of walkers
        burnIn:    int = 100,       # number of burn-in steps
        nsamples:  int = 1000,      # number of samples
        ### Non-CLI arguments ###
        ###  params_dict = {par: value} or {par: [value, min, max]} Sample from [min, max] for walker initialization
        params_dict   = {}, # Dictionary of parameter to override values in cfgInfo if not empty
        moves_mixture = [ emcee.moves.StretchMove() ], # Move mixture for sampler
    ):

    ''' Performs MCMC sampling '''

    ########## SETUP PARAMETERS ############
    ## config files are assumed to contain the MLE values already
    ##   this can be acheived by running fit.py with --seedfile
    ##   and using the include directive in the config file to add the seed file
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

    # print(f'mle_values: {mle_values}')
    # print(f'keys: {keys}')
    # print(f'par_names_flat: {par_names_flat}')
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

        print(f'\n[Burn-in beginning]\n')
        sampler.reset()
        state = sampler.run_mcmc(par_values, burnIn)
        sampler.reset()
        print(f'\n[Burn-in complete. Running sampler]\n')
        sampler.run_mcmc(state, nsamples, progress=True);
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
    MIN_VAL = -1e10
    MAX_VAL =  1e10
    mask = np.logical_and(samples > MIN_VAL, samples < MAX_VAL).all(axis=1)
    samples = samples[mask]
    print(f"Percent samples remaning after masking: {100*samples.shape[0]/(nsamples*nwalkers):0.2f}%")

    ################### COMPUTE MAP ESTIMATE ###################
    map_idx = np.argmax(sampler.get_log_prob(flat=True)) # maximum a posteriori (MAP) location
    map_estimate = samples[map_idx,:]

    return {'samples': samples, 'map_estimate': map_estimate, 'autocorr_time': autocorr_time,
            'acceptance_fraction': acceptance_fraction, 'mle_values': mle_values, 'par_names_flat': par_names_flat,
            'nDim': nDim, 'nwalkers': nwalkers, 'nsamples': nsamples, 'elapsed_fit_time': fit_end_time - fit_start_time,}

################### PLOT RESULTS ###################
def draw_corner(
    results,    # results dictionary from perform_mcmc()
    ofolder,    # output folder name
):
    ''' Draws a corner plot to visualize sampling and parameter correlations '''

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

    ####### DRAW CORNER PLOT #######
    corner_kwargs = {"color": "black", "show_titles": True }
    labels_ReIm = [f'{l.split("::")[-1]}' for l in par_names_flat]
    fig = corner.corner(samples, labels=labels_ReIm, **corner_kwargs)
    nDim = samples.shape[1]
    axes = np.array(fig.axes).reshape((nDim, nDim))
    plot_axvh_fit_params(map_estimate, color='royalblue')
    plot_axvh_fit_params(mle_values, color='tab:green')
    plt.savefig(f"{ofolder}/mcmc.png")

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


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='emcee fitter')
    parser.add_argument('cfgfile', type=str, help='Config file name')
    parser.add_argument('--ofolder', type=str, default='mcmc', help='Output folder name. Default "mcmc"')
    parser.add_argument('--ofile', type=str, default='mcmc.h5', help='Output file name. Default "mcmc.h5"')
    parser.add_argument('--nwalkers', type=int, default=32, help='Number of walkers. Default 32')
    parser.add_argument('--burnin', type=int, default=100, help='Number of burn-in steps. Default 100')
    parser.add_argument('--nsamples', type=int, default=1000, help='Number of samples. Default 1000')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if exists')
    parser.add_argument('--accelerator', type=str, default='', help='Force use of given "accelerator" ~ [gpu, mpi, mpigpu, gpumpi]. Default "" = cpu')

    args = parser.parse_args(sys.argv[1:])

    cfgfile  = args.cfgfile
    ofolder  = args.ofolder
    ofile    = args.ofile
    nwalkers = args.nwalkers
    burnIn   = args.burnin
    nsamples = args.nsamples
    overwrite_ofile = args.overwrite

    print("\n ===================")
    print(f" cfgfile: {cfgfile}")
    print(f" ofolder: {ofolder}")
    print(f" ofile: {ofile}")
    print(f" nwalkers: {nwalkers}")
    print(f" burnIn: {burnIn}")
    print(f" nsamples: {nsamples}")
    print(f" overwrite_ofile: {overwrite_ofile}")
    print(" ===================\n")

    ############## PREPARE FOR SAMPLER ##############
    assert( os.path.exists(cfgfile) ), 'Config file does not exist at specified path'
    if os.path.isfile(f'{ofolder}/{ofile}') and overwrite_ofile:
        os.system(f'rm {ofolder}/{ofile}')
        print("Overwriting existing output file!")
    os.system(f'mkdir -p {ofolder}')

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ############### INITIALIZE MPI IF REQUESTED ###########
    # (Depends on if bash or mpirun/mpiexec called the python program)
    #######################################################
    USE_MPI, USE_GPU = prepare_mpigpu(args.accelerator) # use mpi/gpu if possible or you forced me to with -accelerator flag

    if USE_MPI:
        from mpi4py import rc as mpi4pyrc
        mpi4pyrc.threads = False
        mpi4pyrc.initialize = False
        from mpi4py import MPI
        RANK_MPI = MPI.COMM_WORLD.Get_rank()
        SIZE_MPI = MPI.COMM_WORLD.Get_size()
        print(f'Rank: {RANK_MPI} of {SIZE_MPI}')
        assert( (USE_MPI and (SIZE_MPI > 1)) )
    else:
        RANK_MPI = 0
        SIZE_MPI = 1

    ################### LOAD LIBRARIES ##################
    from atiSetup import *

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
    results = perform_mcmc(ati, LoadParametersSampler,
                           ofolder   = ofolder,
                           ofile     = ofile,
                           nwalkers  = nwalkers,
                           burnIn    = burnIn,
                           nsamples  = nsamples)
    elapsed_fit_time = results['elapsed_fit_time']
    draw_corner(results, ofolder)

    print(f"Fit time: {elapsed_fit_time} seconds")
    print(f"Total time: {time.time() - start_time} seconds")
