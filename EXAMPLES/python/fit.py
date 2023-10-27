#!/sur/bin/env python

import os
import argparse
import time
from datetime import datetime
import random
import sys
from atiSetup import atiSetup

def performFit(
        fitManager,
        seed_file_tag,
        always_save_seed: bool = False,
    ):
    '''
    Performs a single fit

    Args:
        fitManager: MinuitMinimizationManager
        seed_file_tag: Tag to append to output file names

    Returns:
        bFitFailed: Bool indicating if fit failed
        NLL: Negative log likelihood
    '''
    if args.useMinos:
        fitManager.minosMinimization(True)
    else:
        fitManager.migradMinimization()
    if args.hesse:
        fitManager.setEvaluateHessian(True)

    bFitFailed = fitManager.status() != 0 and fitManager.eMatrixStatus() != 3

    if bFitFailed:
        print("ERROR: fit failed use results with caution...")
        NLL = 1e6
    else:
        NLL = ati.likelihood()

    ati.finalizeFit(seed_file_tag)

    if ( args.seedfile is not None and not bFitFailed ) or always_save_seed:
        ati.fitResults().writeSeed(f'{args.seedfile}_{seed_file_tag}.txt')

    return bFitFailed, NLL

############### PERFORM FIT #############
def runFits(
        ati,
        N: int = 0,
        always_save_seed: bool = False,
    ):
    '''
    Performs N randomized fits, if N=0 then a single fit with no randomization is performed

    Args:
        N: Number of randomized fits to perform

    Returns:
        minNLL: Minimum negative log likelihood
    '''

    if (RANK_MPI==0):
        print(f'LIKELIHOOD BEFORE MINIMIZATION: {ati.likelihood()}')
        fitManager: MinuitMinimizationManager = ati.minuitMinimizationManager()
        fitManager.setMaxIterations(args.maxIter)

        if N == 0: # No randomization
            bFitFailed, minNLL = performFit(fitManager, '0', always_save_seed)
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

                bFitFailed, NLL = performFit(fitManager, f'{i}', always_save_seed)
                if not bFitFailed and NLL < minNLL:
                    minNLL = NLL
                    minFitTag = i

                print(f'LIKELIHOOD AFTER MINIMIZATION: {NLL}\n')

            if minFitTag < 0:
                print("ALL FITS FAILED!")
            else:
                print(f'MINIMUM LIKELHOOD FROM ITERATION {minFitTag} of {N} RANDOM PRODUCTION PARS = {minNLL}')
                os.system(f'cp {fitName}_{minFitTag}.fit {fitName}.fit')
                if args.seedfile is not None:
                    os.system(f'cp {args.seedfile}_{minFitTag}.txt {args.seedfile}.txt')

    if USE_MPI:
       ati.exitMPI()

    return minNLL

if __name__ == '__main__':
    start_time = time.time()

    ############## PARSE COMMANDLINE ARGUMENTS #############
    parser = argparse.ArgumentParser(description="Perform MLE fits")
    rndSeed = random.seed(datetime.now().timestamp())
    parser.add_argument('cfgfile',       type=str,                  help='AmpTools Configuration file')
    parser.add_argument('--seedfile',    type=str, default=None,    help='Output file for seeding next fit based on this fit. Do not include extension')
    parser.add_argument('--always_save_seed', action='store_true',  help='Always save the seed file, even if the fit fails')
    parser.add_argument('--numRnd',      type=int, default=0,       help='Perform N fits each seeded with random parameters')
    parser.add_argument('--randomSeed',  type=int, default=rndSeed, help='Sets the random seed used by the random number generator for the fits with randomized initial parameters. If not set, will use the current time.')
    parser.add_argument('--maxIter',     type=int, default=100000,  help='Maximum number of fit iterations')
    parser.add_argument('--useMinos',    action='store_true',       help='Use MINOS instead of MIGRAD')
    parser.add_argument('--hesse',       action='store_true',       help='Evaluate HESSE matrix after minimization')
    parser.add_argument('--scanPar',     type=str, default=None,    help='Perform a scan of the given parameter. Stepsize, min, max are to be set in the config file')
    parser.add_argument('--accelerator', type=str, default='',      help='Force use of given "accelerator" ~ [gpu, mpi, mpigpu, gpumpi]')

    args = parser.parse_args(sys.argv[1:])
    if args.randomSeed is None:
        args.randomSeed = int(time.time())

    cfgfile = args.cfgfile
    assert( os.path.isfile(cfgfile) ), f'Config file does not exist at specified path'

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    USE_MPI, USE_GPU, RANK_MPI = atiSetup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE ##############
    if RANK_MPI == 0:
        print("\n\n === COMMANDLINE ARGUMENTS === ")
        print("Config file:", args.cfgfile)
        print("Seed file:", args.seedfile)
        print("Always save seed:", args.always_save_seed)
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
    AmpToolsInterface.registerAmplitude( BreitWigner() )
    AmpToolsInterface.registerAmplitude( Piecewise() )
    AmpToolsInterface.registerAmplitude( PhaseOffset() )
    AmpToolsInterface.registerAmplitude( TwoPiAngles() )
    AmpToolsInterface.registerDataReader( DataReader() )
    AmpToolsInterface.registerDataReader( DataReaderFilter() )
    AmpToolsInterface.registerDataReader( DataReaderBootstrap() )

    ati = AmpToolsInterface( cfgInfo )

    AmpToolsInterface.setRandomSeed(args.randomSeed)

    fit_start_time = time.time()
    nll = runFits(ati, args.numRnd, args.always_save_seed)

    print("\nDone! MPI.Finalize() / MPI.Init() automatically called at script end / start\n") if USE_MPI else print("\nDone!")
    print(f"Fit time: {time.time() - fit_start_time} seconds")
    print(f"Total time: {time.time() - start_time} seconds")
    print(f"Final Likelihood: {nll}") # Need this for for unit-tests
