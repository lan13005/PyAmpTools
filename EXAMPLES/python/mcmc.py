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
from utils import prepare_mpigpu

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
args = parser.parse_args()

cfgfile = args.cfgfile
ofolder = args.ofolder
ofile = args.ofile
overwrite_ofile = args.overwrite
NWALKERS = args.nwalkers
BURN_IN = args.burnin
NSAMPLES = args.nsamples

print("\n ===================")
print(f" cfgfile: {cfgfile}")
print(f" ofolder: {ofolder}")
print(f" ofile: {ofile}")
print(f" NWALKERS: {NWALKERS}")
print(f" BURN_IN: {BURN_IN}")
print(f" NSAMPLES: {NSAMPLES}")
print(f" overwrite_ofile: {overwrite_ofile}")
print(" ===================\n")

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

############## UTILITY FUNCTIONS ##############
def LogProb(
    par_values:  npt.NDArray[np.float64] = np.array([]), # [real1, imag1, real2, imag2, ...]
    ):
    ''' Log probability function = Log Likelihood if no prior '''

    ## Calculate Log likelihood
    ll = -1e7
    parameters = LoadParameters.unflatten_parameters(par_values, KEY)
    for name, value in parameters.items():
        parMgr[name] = value
    ll = -ati.likelihood()

    ## Add lasso prior on parameter values we know are small
    # lasso = 0.1
    prior = 0 # -lasso * np.sum(np.abs(par_values[PAR_INDICES >= 2]))

    log_prob = ll + prior
    # print(f'LogProb: {log_prob} = {ll} + {prior}')
    return log_prob

############### LOAD INTO MANAGER ###############
LoadParameters = LoadParameters()
LoadParameters.load_cfg( cfgInfo )

# SELECTED_VALUES = {'a2mass': 1.31790479736726, 'a2width': 0.130376506768682}
# MLE_VALUES, KEY, PAR_NAMES_FLAT  = LoadParameters.flatten_parameters(SELECTED_VALUES)
MLE_VALUES, KEY, PAR_NAMES_FLAT  = LoadParameters.flatten_parameters()

if NWALKERS < 2*len(MLE_VALUES):
    print("Number of walkers must be at least twice the number of parameters. Overwriting NWALKERS to 2*len(MLE_VALUES)\n")
    NWALKERS = 2*len(MLE_VALUES)

print(f'MLE_VALUES: {MLE_VALUES}')
print(f'KEY: {KEY}')
print(f'PAR_NAMES_FLAT: {PAR_NAMES_FLAT}')
NDIM = len(MLE_VALUES)


############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
fit_start_time = time.time()
if not os.path.exists(f'{ofolder}/{ofile}'):
    ############ INITIALIZE ATI ############
    ati = AmpToolsInterface( cfgInfo )
    parMgr: ParameterManager = ati.parameterManager()

    print(f' ================== RUNNING MCMC ================== ')
    # Initialize walkers in an N-ball around the MLE estimate
    par_values = np.array(MLE_VALUES)
    par_values = np.repeat(par_values, NWALKERS).reshape(NDIM, NWALKERS).T
    par_values *= ( 1 + 0.01 * np.random.normal(0, 1, size=(NWALKERS, NDIM)) )

    backend = emcee.backends.HDFBackend(f'{ofolder}/{ofile}')
    backend.reset(NWALKERS, NDIM)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, LogProb, backend=backend)

    state = sampler.run_mcmc(par_values, BURN_IN)
    sampler.reset()
    sampler.run_mcmc(state, NSAMPLES, progress=True);
    acceptance_fraction = np.mean(sampler.acceptance_fraction) # function implementation in github, acceptance_fraction not available from HDF5 backend
    autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
    print(f"Mean acceptance fraction: {0:.3f}".format(acceptance_fraction))
    print(f"Autocorrelation time: {0:.3f} steps".format(autocorr_time))

else:
    print(f' ================== LOADING MCMC ================== ')
    sampler = emcee.backends.HDFBackend(f'{ofolder}/{ofile}')
fit_end_time = time.time()

################# CALCULATE AUTOCORR AND ACCEPTANCE FRAC #################
analysis_start_time = time.time()
samples = sampler.get_chain(flat=True) # returns (NSAMPLES*NWALKERS, 5) flattened array
if isinstance(sampler, emcee.EnsembleSampler):
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
else: # HDF5 backend
    acceptance_fraction = np.mean(sampler.accepted / sampler.iteration) # function implementation in github, acceptance_fraction not available from HDF5 backend
autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
print(f"Mean acceptance fraction: {0:.3f}".format(acceptance_fraction))
print(f"Autocorrelation time: {0:.3f} steps".format(autocorr_time))
MIN_VAL = -1e10
MAX_VAL = 1e10
mask = np.logical_and(samples > MIN_VAL, samples < MAX_VAL).all(axis=1)
samples = samples[mask]
print(f"Percent samples remaning after masking: {100*samples.shape[0]/(NSAMPLES*NWALKERS):0.2f}%")

################### COMPUTE MAP ESTIMATE ###################
map_idx = np.argmax(sampler.get_log_prob(flat=True)) # maximum a posteriori (MAP) location
map_estimate = samples[map_idx,:]

################### PLOT RESULTS ###################
def plot_axvh_fit_params(values, color='black'):
    axes = np.array(fig.axes).reshape((NDIM, NDIM))
    for yi in range(NDIM):
        for xi in range(yi):
            axes[yi, xi].axvline(values[xi], color=color, linewidth=4, alpha=0.8)
            axes[yi, xi].axhline(values[yi], color=color, linewidth=4, alpha=0.8)
        axes[yi, yi].axvline(values[yi], color=color, linewidth=4, alpha=0.8)
corner_kwargs = {"color": "black", "show_titles": True }
labels_ReIm = [f'{l.split("::")[-1]}' for l in PAR_NAMES_FLAT]
fig = corner.corner(samples, labels=labels_ReIm, **corner_kwargs)
plot_axvh_fit_params(map_estimate, color='royalblue')
plot_axvh_fit_params(MLE_VALUES, color='tab:green')
plt.savefig(f"{ofolder}/mcmc.png")


print(' =================== RESULTS =================== ')
print(f'MAP Estimates from {len(samples)} samples obtained over {NWALKERS} walkers:')
[print(f'   {l:20} = {v:0.3f}') for l, v in zip(PAR_NAMES_FLAT, map_estimate)]
print(' =============================================== ')

# import pygtc
# GTC_ReIM = pygtc.plotGTC(chains=[samples], paramNames=PAR_NAMES_FLAT,
#                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page', plotName=f'{ofolder}/mcmc_ReIm.pdf', nContourLevels=3)
# GTC = pygtc.plotGTC(chains=[intensity], paramNames=PAR_NAMES,
#                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page', plotName=f'{ofolder}/mcmc.pdf', nContourLevels=3)

print(f"Fit time: {fit_end_time - fit_start_time} seconds")
print(f"Analysis (calc autocorr, acceptance fractions, drawing results, ...) time: {time.time() - analysis_start_time} seconds")
print(f"Total time: {time.time() - start_time} seconds")
