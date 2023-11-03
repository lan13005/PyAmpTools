#!/sur/bin/env python

import os
import argparse
import time
from datetime import datetime
import random
import sys
import atiSetup

def performFit(
        ati,
        seed_file_tag,
        seedfile: str = "seed",
        useMinos: bool = False,
        hesse: bool = False,
    ):
    '''
    Performs a single fit

    Args:
        ati (AmpToolsInterface):  AmpToolsInterface instance
        seed_file_tag (str): Tag to append to output file names to distinguish multiple runs

    Returns:
        bFitFailed (bool): flag indicating if fit failed
        NLL (double): Negative log likelihood
    '''
    fitManager: MinuitMinimizationManager = ati.minuitMinimizationManager()

    if useMinos:
        fitManager.minosMinimization(True)
    else:
        fitManager.migradMinimization()
    if hesse:
        fitManager.setEvaluateHessian(True)

    bFitFailed = fitManager.status() != 0 and fitManager.eMatrixStatus() != 3

    if bFitFailed:
        print("ERROR: fit failed use results with caution...")
        NLL = 1e6
    else:
        NLL = ati.likelihood()

    ati.finalizeFit(seed_file_tag)

    if seedfile is not None:
        ati.fitResults().writeSeed(f'{seedfile}_{seed_file_tag}.txt')

    return bFitFailed, NLL

############### PERFORM FIT #############
def runFits(
        ati,
        N: int = 0,
        RANK_MPI: int = 0,
        maxIter = 100000,
        USE_MPI: bool = False,
        seedfile: str = "seed",
        useMinos: bool = False,
        hesse: bool = False,
    ):
    '''
    Performs N randomized fits by calling performFit(), if N=0 then a single fit with no randomization is performed

    Args:
        ati (AmpToolsInterface): AmpToolsInterface instance
        N (int): Number of randomized fits to perform
        RANK_MPI (int): MPI rank. Default to 0 for non-MPI.
        maxIter (int): Maximum number of iterations. Default to 100000.
        USE_MPI (bool): Use MPI. Default to False.
        seedfile (str): Output file for seeding next fit based on this fit. Default to "seed" name prefix.
        useMinos (bool): Use MINOS instead of MIGRAD. Default to False.
        hesse (bool): Evaluate HESSE matrix after minimization. Default to False.

    Returns:
        minNLL (double): Minimum negative log likelihood
    '''

    fitargs = (seedfile, useMinos, hesse)

    minNLL = 1e6
    if (RANK_MPI==0):
        print(f'LIKELIHOOD BEFORE MINIMIZATION: {ati.likelihood()}')
        fitManager: MinuitMinimizationManager = ati.minuitMinimizationManager()
        fitManager.setMaxIterations(maxIter)

        if N == 0: # No randomization
            bFitFailed, minNLL = performFit( ati, '0', *fitargs )
            print(f'LIKELIHOOD AFTER MINIMIZATION (NO RANDOMIZATION): {minNLL}')

        else: # Randomized parameters
            fitName     = cfgInfo.fitName()
            maxFraction = 0.5
            minNLL       = sys.float_info.max
            minFitTag   = -1

            parRangeKeywords = cfgInfo.userKeywordArguments("parRange")

            for i in range(N):
                print("\n###############################################")
                print(f'#############   FIT {i} OF {N} ###############')
                print("###############################################\n")

                ati.reinitializePars()
                ati.randomizeProductionPars(maxFraction)
                for ipar in range(len(parRangeKeywords)):
                    ati.randomizeParameter(parRangeKeywords[ipar][0], float(parRangeKeywords[ipar][1]), float(parRangeKeywords[ipar][2]))

                bFitFailed, NLL = performFit( ati, f'{i}', *fitargs )
                if not bFitFailed and NLL < minNLL:
                    minNLL = NLL
                    minFitTag = i

                print(f'LIKELIHOOD AFTER MINIMIZATION: {NLL}\n')

            if minFitTag < 0:
                print("ALL FITS FAILED!")
            else:
                print(f'MINIMUM LIKELHOOD FROM ITERATION {minFitTag} of {N} RANDOM PRODUCTION PARS = {minNLL}')
                os.system(f'cp {fitName}_{minFitTag}.fit {fitName}.fit')
                if seedfile is not None:
                    os.system(f'cp {seedfile}_{minFitTag}.txt {seedfile}.txt')

    if USE_MPI:
       ati.exitMPI()

    return minNLL

if __name__ == '__main__':
    start_time = time.time()

    ############## PARSE COMMANDLINE ARGUMENTS #############
    parser = argparse.ArgumentParser(description="Perform MLE fits")
    rndSeed = random.seed(datetime.now().timestamp())
    parser.add_argument('cfgfile',       type=str,                   help='AmpTools Configuration file')
    parser.add_argument('--seedfile',    type=str, default="seed",   help='Output file for seeding next fit based on this fit. Do not include extension')
    parser.add_argument('--numRnd',      type=int, default=0,        help='Perform N fits each seeded with random parameters')
    parser.add_argument('--randomSeed',  type=int, default=rndSeed,  help='Sets the random seed used by the random number generator for the fits with randomized initial parameters. If not set, will use the current time.')
    parser.add_argument('--maxIter',     type=int, default=100000,   help='Maximum number of fit iterations')
    parser.add_argument('--useMinos',    action='store_true',        help='Use MINOS instead of MIGRAD')
    parser.add_argument('--hesse',       action='store_true',        help='Evaluate HESSE matrix after minimization')
    parser.add_argument('--scanPar',     type=str, default=None,     help='Perform a scan of the given parameter. Stepsize, min, max are to be set in the config file')
    parser.add_argument('--accelerator', type=str, default='mpigpu', help='Use accelerator if available ~ [cpu, gpu, mpi, mpigpu, gpumpi]')

    args = parser.parse_args(sys.argv[1:])
    if args.randomSeed is None:
        args.randomSeed = int(time.time())

    cfgfile = args.cfgfile
    assert( os.path.isfile(cfgfile) ), f'Config file does not exist at specified path'

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE ##############
    if RANK_MPI == 0:
        print("\n\n === COMMANDLINE ARGUMENTS === ")
        print("Config file:", args.cfgfile)
        print("Seed file:", args.seedfile)
        print("Number of random fits:", args.numRnd)
        print("Random seed:", args.randomSeed)
        print("Maximum iterations:", args.maxIter)
        print("Use MINOS:", args.useMinos)
        print("Evaluate HESSE matrix:", args.hesse)
        print("Scanning Parameter:", args.scanPar)
        print(" ============================= \n\n")

    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
    if RANK_MPI == 0:
        cfgInfo.display()

    # ############## REGISTER OBJECTS FOR AMPTOOLS ##############
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
    AmpToolsInterface.registerDataReader( DataReaderBootstrap() )

    ati = AmpToolsInterface( cfgInfo )

    AmpToolsInterface.setRandomSeed(args.randomSeed)

    fit_start_time = time.time()
    nll = runFits( ati, N = args.numRnd, \
                    seedfile = args.seedfile, \
                    RANK_MPI = RANK_MPI, \
                    useMinos = args.useMinos, \
                    hesse = args.hesse, \
                    maxIter = args.maxIter, \
                    USE_MPI = USE_MPI )

    if RANK_MPI == 0:
        print("\nDone! MPI.Finalize() / MPI.Init() automatically called at script end / start\n") if USE_MPI else print("\nDone!")
        print(f"Fit time: {time.time() - fit_start_time} seconds")
        print(f"Total time: {time.time() - start_time} seconds")
        print(f"Final Likelihood: {nll}") # Need this for for unit-tests
