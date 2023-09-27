import numpy as np
import numpy.typing as npt
import emcee
import ROOT
import os
from typing import List
import matplotlib.pyplot as plt
from amploader import load_amplitude_info, flatten_amplitude_parts, collect_amplitude_parts
import corner

############## SET OUTPUT FOLDER ##############
ofolder = 'mcmc/Lasso' # output results here
save_model = "mcmc.h5" # save MCMC results here
os.system(f'mkdir -p {ofolder}')
NWALKERS = int(32)
BURN_IN = int(100)
NSAMPLES = int(1e3)

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']

############## LOAD LIBRARIES ##############
ROOT.gSystem.Load('libAmps.so')
ROOT.gSystem.Load('libDataIO.so')
ROOT.gSystem.Load('libAmpTools.so')

# Dummy functions that just prints initialization
#  This is to make sure the libraries are loaded
#  as python is interpreted
ROOT.initializeAmps(True)
ROOT.initializeDataIO(True)

################ SET ALIAS ###################
ConfigFileParser  = ROOT.ConfigFileParser
ConfigurationInfo = ROOT.ConfigurationInfo
AmpToolsInterface = ROOT.AmpToolsInterface
Zlm               = ROOT.Zlm
ROOTDataReader    = ROOT.ROOTDataReader
ParameterManager  = ROOT.ParameterManager
AmplitudeInfo     = ROOT.AmplitudeInfo

############## LOAD CONFIGURATION FILE ##############
cfgfile = f'{REPO_HOME}/gen_amp/fit_res.cfg'
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


############## FIND AVAILABLE AMPS ##############
# amplitudes: AmplitudeInfo = cfgInfo.amplitudeList()
# amp_names = []
# PAR_NAMES = []
# for amp in amplitudes:
#     if amp.fixed(): continue
#     full_name = amp.fullName() # "Reaction::Sum::Amplitude"
#     amp_name = full_name.split("::")[-1] # Extract Amplitude from full name
#     if amp_name in amp_names: continue
#     amp_names.append(amp_name)
#     PAR_NAMES.append(full_name)
# PAR_NAMES = np.array(PAR_NAMES)
# print(f'amp_names: {amp_names}')
# print(f'PAR_NAMES: {PAR_NAMES}')


############### LOAD MLE ESTIMATE ###############
mle_fit = "/w/halld-scshelf2101/lng/WORK/PyAmpTools/gen_amp/MLE.fit" # seed file from amptools fit
PAR_NAMES, par_values = load_amplitude_info(mle_fit)
MLE_VALUES, PAR_NAMES_PARTS, PAR_INDICES = flatten_amplitude_parts(PAR_NAMES, par_values)
NDIM = len(MLE_VALUES) # Dimensionality of parameter space

############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
if not os.path.exists(f'{ofolder}/mcmc.h5'):
    print(f' ================== RUNNING MCMC ================== ')
    par_values = np.array(MLE_VALUES)
    par_values = np.repeat(par_values, NWALKERS).reshape(NDIM, NWALKERS).T
    par_values *= ( 1 + 0.01 * np.random.normal(0, 1, size=(NWALKERS, NDIM)) )

    backend = emcee.backends.HDFBackend(f'{ofolder}/{save_model}')
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
    sampler = emcee.backends.HDFBackend(f'{ofolder}/{save_model}')


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
