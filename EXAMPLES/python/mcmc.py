import numpy as np
import numpy.typing as npt
import emcee
import ROOT
import os
from typing import List
import matplotlib.pyplot as plt
from amploader import load_amplitude_info, flatten_amplitude_parts, collect_amplitude_parts
import corner
import argparse

parser = argparse.ArgumentParser(description='emcee fitter')
parser.add_argument('-c', type=str, help='Config file name')
parser.add_argument('-m', type=str, help='MLE fitted seed file name')
parser.add_argument('-o', type=str, default='mcmc', help='Output folder name')
parser.add_argument('-f', type=str, default='mcmc.h5', help='Output file name')
parser.add_argument('-n', type=int, default=32, help='Number of walkers')
parser.add_argument('-b', type=int, default=100, help='Number of burn-in steps')
parser.add_argument('-s', type=int, default=1000, help='Number of samples')
args = parser.parse_args()

cfgfile = args.c
mle_fit = args.m # seed file from amptools fit
ofolder = args.o
ofile = args.f
NWALKERS = args.n
BURN_IN = args.b
NSAMPLES = args.s
os.system(f'mkdir -p {ofolder}')

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']
os.environ['ATI_USE_MPI'] = "0" # set to 1 to use MPI libraries
from atiSetup import *

############## LOAD CONFIGURATION FILE ##############
parser = ConfigFileParser(cfgfile)
cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
cfgInfo.display()

############## REGISTER OBJECTS FOR AMPTOOLS ##############
AmpToolsInterface.registerAmplitude( Zlm() )
AmpToolsInterface.registerDataReader( ROOTDataReader() )
ati = AmpToolsInterface( cfgInfo )
parMgr: ParameterManager = ati.parameterManager()

############## UTILITY FUNCTIONS ##############
def LogProb(
    par_values:  npt.NDArray[np.float64] = np.array([]), # [real1, imag1, real2, imag2, ...]
    PAR_NAMES:   npt.NDArray[np.str_]    = np.array([]),
    PAR_INDICES: npt.NDArray[np.int64]   = np.array([])
    ):
    ''' Log probability function = Log Likelihood if no prior '''

    ## Calculate Log likelihood
    ll = -1e7
    complex_values = collect_amplitude_parts(par_values, PAR_NAMES, PAR_INDICES)
    for name, complex_value in zip(PAR_NAMES, complex_values):
        # print(f'Setting {name} to {complex_value}')
        parMgr.setProductionParameter(name, complex_value)
    ll = -ati.likelihood()

    ## Add lasso prior on parameter values we know are small
    lasso = 0.1
    prior = -lasso * np.sum(np.abs(par_values[PAR_INDICES >= 2]))

    log_prob = ll + prior
    # print(f'LogProb: {log_prob} = {ll} + {prior}')
    return log_prob


############### LOAD MLE ESTIMATE ###############
PAR_NAMES, par_values = load_amplitude_info(mle_fit)
MLE_VALUES, PAR_NAMES_PARTS, PAR_INDICES = flatten_amplitude_parts(PAR_NAMES, par_values)
NDIM = len(MLE_VALUES) # Dimensionality of parameter space

############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
if not os.path.exists(f'{ofolder}/mcmc.h5') or os.path.getsize(f'{ofolder}/mcmc.h5') < 20000:
    print(f' ================== RUNNING MCMC ================== ')
    par_values = np.array(MLE_VALUES)
    par_values = np.repeat(par_values, NWALKERS).reshape(NDIM, NWALKERS).T
    par_values *= ( 1 + 0.01 * np.random.normal(0, 1, size=(NWALKERS, NDIM)) )

    backend = emcee.backends.HDFBackend(f'{ofolder}/{ofile}')
    backend.reset(NWALKERS, NDIM)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, LogProb, args=[PAR_NAMES, PAR_INDICES], backend=backend)

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


################# CALCULATE AUTOCORR AND ACCEPTANCE FRAC #################
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
map_estimate_ReIm = samples[map_idx,:]
intensity = np.empty((samples.shape[0], len(PAR_NAMES)))
for i in range(len(PAR_NAMES)):
    real = samples[:, np.argwhere(PAR_INDICES==(2*i  ))[0][0]]   if 2*i in PAR_INDICES else np.zeros(samples.shape[0])
    imag = samples[:, np.argwhere(PAR_INDICES==(2*i+1))[0][0]] if 2*i+1 in PAR_INDICES else np.zeros(samples.shape[0])
    intensity[:, i] = real**2 + imag**2
map_estimate = intensity[map_idx,:]


################### PLOT RESULTS ###################
corner_kwargs = {"color": "royalblue", "truth_color": "orange", "show_titles": True }
labels_ReIm = [f'{l.split("::")[-1]}' for l in PAR_NAMES_PARTS]
fig = corner.corner(samples, labels=labels_ReIm, truths=map_estimate_ReIm, **corner_kwargs)
plt.savefig(f"{ofolder}/mcmc_ReIm.png")
labels = [f'{l.split("::")[-1]}' for l in PAR_NAMES]
fig = corner.corner(intensity, labels=labels, truths=map_estimate, **corner_kwargs)
plt.savefig(f"{ofolder}/mcmc.png")


print(' =================== RESULTS =================== ')
print(f'MAP for ReIm parts: ')
[print(f'   {l}= {v:0.3f}') for l, v in zip(PAR_NAMES_PARTS, map_estimate_ReIm)]
print(f'MAP for Intensity: {map_estimate}')
[print(f'   {l}= {v:0.3f}') for l, v in zip(PAR_NAMES, map_estimate)]
print(' =============================================== ')

# import pygtc
# GTC_ReIM = pygtc.plotGTC(chains=[samples], paramNames=PAR_NAMES_PARTS,
#                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page', plotName=f'{ofolder}/mcmc_ReIm.pdf', nContourLevels=3)
# GTC = pygtc.plotGTC(chains=[intensity], paramNames=PAR_NAMES,
#                     chainLabels=['MCMC samples'], legendMarker='Auto', figureSize='MNRAS_page', plotName=f'{ofolder}/mcmc.pdf', nContourLevels=3)
