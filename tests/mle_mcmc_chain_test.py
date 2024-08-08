import numpy as np
import subprocess
import os

FIT_CONFIG = "sdme.cfg"
FIT_MCMC_CONFIG = "sdme.cfg"
MCMC_KWARGS = "--nwalkers 32 --nsamples 100 --corner_ofile mcmc/corner.png --corner_format intensity"


def exec_mle():
    assert os.path.exists(FIT_CONFIG), "Config file does not exist at specified path"
    result = subprocess.run(["pa", "fit", FIT_CONFIG], capture_output=True, text=True)
    seed_exists = os.path.exists("seed_0.txt")
    result_exists = os.path.exists("result_0.fit")
    print("\n\n========================== MLE ==========================")
    print(result)
    print("========================== MLE ==========================\n\n")
    assert seed_exists and result_exists, "MLE fit failed to produce seed and fit results file"


def check_mle():
    fit_sdmes = {}  # Fitted SDME parameters
    with open("seed_0.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("parameter"):
                par, value = line.split()[1:3]
                fit_sdmes[par] = float(value)

    gen_sdmes = {}  # Generated SDME parameters
    with open("samples/sdme.cfg", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("define"):
                line = line.replace("\\s+", " ")
                par, value = line.split()[1:3]
                gen_sdmes[par] = float(value)

    print("\n======================================")
    print("PARAMETER | FITTED | GENERATED")
    mae = 0
    for par in fit_sdmes:
        print(f"{par:<10}| {fit_sdmes[par]:<6.3f} | {gen_sdmes[par]:0.3f}")
        mae += np.abs(fit_sdmes[par] - gen_sdmes[par])
    mae /= len(fit_sdmes)

    print(f"\nMean absolute error: {mae:0.4f}")
    print("======================================\n")

    assert mae < 0.01, "Mean absolute error is too high! Above my required threshold of 0.01."


def exec_mcmc():
    assert os.path.exists(FIT_MCMC_CONFIG), "Config file does not exist at specified path"
    result = subprocess.run(["pa", "mcmc", FIT_MCMC_CONFIG] + MCMC_KWARGS.split(" "), capture_output=True, text=True)
    print("\n\n========================== MCMC ==========================")
    print(result)
    print("========================== MCMC ==========================\n\n")
    assert os.path.exists("mcmc/emcee_state.h5"), "MCMC fit failed to produce mcmc.h5 file"
    assert os.path.exists("mcmc/corner.png"), "MCMC fit failed to produce corner.png file"


def test_mle_mcmc_chain():
    if not os.path.exists("tests/samples/SDME_EXAMPLE"):
        print("Skipping MLE-MCMC chain test. SDME_EXAMPLE directory not found.")
        return
    os.chdir("tests/samples/SDME_EXAMPLE")

    ## Perform MLE fit
    exec_mle()
    check_mle()

    ## Create MCMC config file (same as amptools cfg but seeded with MLE)
    os.system("cp sdme.cfg sdme_mcmc.cfg")
    dump = "\ninclude seed_0.txt\n"
    with open("sdme_mcmc.cfg", "a") as f:
        f.write(dump)

    ## Perform MCMC fit
    exec_mcmc()

    ## Clean up files
    # os.system(f'rm -f seed_0.txt result_0.fit')
    # os.system(f'rm -f mcmc sdme_mcmc.cfg')

    os.chdir("../../..")


# def run_mcmc(nametag):

#     ''' Run MCMC fits '''

#     from pyamptools import atiSetup
#     USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), use_genamp=True) # RANK_MPI defaults to 0 even if not using MPI

#     ofolder = nametag
#     nwalkers = 32
#     burnIn = 0
#     nsamples = 200
#     seed = 42 # for reproducibility

#     cfgfile = f'{ofolder}/mcmc.cfg'
#     ofile = 'mcmc.h5'
#     corner_ofile = 'corner.png'

#     ############## PREPARE FOR SAMPLER ##############
#     assert( os.path.exists(cfgfile) ), 'Config file does not exist at specified path'
#     os.system(f'mkdir -p {ofolder}')

#     ############## LOAD CONFIGURATION FILE ##############
#     parser = ConfigFileParser(cfgfile)
#     cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
#     cfgInfo.display()

#     ############## REGISTER OBJECTS FOR AMPTOOLS ##############
#     AmpToolsInterface.registerAmplitude( Zlm() )
#     AmpToolsInterface.registerDataReader( DataReader() )
#     AmpToolsInterface.registerAmplitude( Vec_ps_refl() )
#     AmpToolsInterface.registerAmplitude( OmegaDalitz() )
#     AmpToolsInterface.registerDataReader( DataReaderTEM() )

#     ati = AmpToolsInterface( cfgInfo )

#     LoadParametersSampler = LoadParameters(cfgInfo)

#     ############## RUN MCMC ##############
#     np.random.seed(seed)

#     atis = [ati] # list of AmpToolsInterface objects to run MCMC on
#     LoadParametersSamplers = [LoadParametersSampler] # list of LoadParameters objects to run MCMC on

#     [ ati.parameterManager().setDoCovarianceUpdate(False) for ati in atis ] # No internal Minuit fit = no covariance matrix

#     mcmcMgr = mcmcManager(atis, LoadParametersSamplers, f'{ofolder}/{ofile}')

#     mcmcMgr.perform_mcmc(
#         nwalkers = nwalkers,
#         burnIn   = burnIn,
#         nsamples = nsamples,
#         sampler_kwargs = {'progress': False},  # turn off progress bar for cleaner output
#     )

#     print(f"Fit time: {mcmcMgr.elapsed_fit_time} seconds")

#     mcmcMgr.draw_corner(f'{ofolder}/{corner_ofile}', format='fitfrac' )
