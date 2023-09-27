import ROOT
import os
import argparse
import time
from datetime import datetime
import random
import sys
from utils import get_pid_family

def performFit(fitManager, seed_file_tag):
    ''' Performs a single fit '''
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

    if args.seedfile is not None and not bFitFailed:
        ati.fitResults().writeSeed(f'{args.seedfile}_{seed_file_tag}.txt')

    return bFitFailed, NLL

############### PERFORM FIT #############
def runFits( N: int = 0 ):
    '''
    Performs N randomized fits, if N=0 then a single fit with no randomization is performed
    '''

    if (RANK_MPI==0):
        print(f'LIKELIHOOD BEFORE MINIMIZATION: {ati.likelihood()}')
        fitManager: MinuitMinimizationManager = ati.minuitMinimizationManager()
        fitManager.setMaxIterations(args.maxIter)

        if N == 0: # No randomization
            bFitFailed, NLL = performFit(fitManager, '0')
            print(f'LIKELIHOOD AFTER MINIMIZATION (NO RANDOMIZATION): {NLL}')

        else: # Randomized parameters
            fitName     = cfgInfo.fitName()
            maxFraction = 0.5
            minLL       = sys.float_info.max
            minFitTag   = -1

            parRangeKeywords = cfgInfo.userKeywordArguments("parRange")

            for i in range(N):
                print("###############################################")
                print(f'#############   FIT {i} OF {N} ###############')
                print("###############################################")

                ati.reinitializePars()
                ati.randomizeProductionPars(maxFraction)
                for ipar in range(len(parRangeKeywords)):
                    ati.randomizeParameter(parRangeKeywords[ipar][0], float(parRangeKeywords[ipar][1]), float(parRangeKeywords[ipar][2]))

                bFitFailed, NLL = performFit(fitManager, f'{i}')
                if not bFitFailed and NLL < minLL:
                    minLL = NLL
                    minFitTag = i

                print(f'LIKELIHOOD AFTER MINIMIZATION: {NLL}')

            if minFitTag < 0:
                print("ALL FITS FAILED!")
            else:
                print(f'MINIMUM LIKELHOOD FROM ITERATION {minFitTag} of {N} RANDOM PRODUCTION PARS = {minLL}')
                os.system(f'cp {fitName}_{minFitTag}.fit {fitName}.fit')
                if args.seedfile is not None:
                    os.system(f'cp {args.seedfile}_{minFitTag}.txt {args.seedfile}.txt')

    if USE_MPI:
        ati.exitMPI()

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']

#################### INITIALIZE MPI IF REQUESTED ###################
from mpi4py import rc as mpi4pyrc
mpi4pyrc.threads = False
from mpi4py import MPI
import sys
RANK_MPI = MPI.COMM_WORLD.Get_rank()
SIZE_MPI = MPI.COMM_WORLD.Get_size()
caller, parent = get_pid_family()
SUFFIX, USE_MPI = ("_MPI", True) if "mpi" in parent else ("", False) 
assert( (USE_MPI and (SIZE_MPI > 1)) or not USE_MPI )
if USE_MPI:
    print(f'Rank: {RANK_MPI} of {SIZE_MPI}')

############## PARSE COMMANDLINE ARGUMENTS #############
parser = argparse.ArgumentParser(description="Perform MLE fits")
rndSeed = random.seed(datetime.now().timestamp())
parser.add_argument('-c', '--config',     type=str,                  help='AmpTools Configuration file')
parser.add_argument('-s', '--seedfile',   type=str, default=None,    help='Output file for seeding next fit based on this fit')
parser.add_argument('-r', '--numRnd',     type=int, default=0,       help='Perform N fits each seeded with random parameters')
parser.add_argument('-rs','--randomSeed', type=int, default=rndSeed, help='Sets the random seed used by the random number generator for the fits with randomized initial parameters. If not set, will use the current time.')
parser.add_argument('-m', '--maxIter',    type=int, default=10000,   help='Maximum number of fit iterations')
parser.add_argument('-n', '--useMinos',   action='store_true',       help='Use MINOS instead of MIGRAD')
parser.add_argument('-H', '--hesse',      action='store_true',       help='Evaluate HESSE matrix after minimization')
parser.add_argument('-p', '--scanPar',    type=str, default=None,    help='Perform a scan of the given parameter. Stepsize, min, max are to be set in the config file')

args = parser.parse_args()
if args.randomSeed is None:
    args.randomSeed = int(time.time())
if args.config is None:
    print("No config file specified")
    exit(1)

if RANK_MPI == 0:
    print("\n\n === COMMANDLINE ARGUMENTS === ")
    print("Config file:", args.config)
    print("Seed file:", args.seedfile)
    print("Number of random fits:", args.numRnd)
    print("Random seed:", args.randomSeed)
    print("Maximum iterations:", args.maxIter)
    print("Use MINOS:", args.useMinos)
    print("Evaluate HESSE matrix:", args.hesse)
    print("Scanning Parameter:", args.scanPar)
    print(" ============================= \n\n")


#################### LOAD LIBRARIES ###################
ROOT.gSystem.Load(f'libAmpTools{SUFFIX}.so')
ROOT.gSystem.Load(f'libDataIO.so')
ROOT.gSystem.Load(f'libAmps.so')
if RANK_MPI == 0:
    print(f'Loaded libraries: libAmpTools{SUFFIX}.so, libDataIO.so, libAmps.so')

# Dummy functions that just prints "initialization"
#  This is to make sure the libraries are loaded
#  as python is interpreted.
ROOT.initializeAmps(True)   if RANK_MPI == 0 else ROOT.initializeAmps(False)
ROOT.initializeDataIO(True) if RANK_MPI == 0 else ROOT.initializeDataIO(False)

##################### SET ALIAS ########################
ConfigFileParser            = ROOT.ConfigFileParser
ConfigurationInfo           = ROOT.ConfigurationInfo
if USE_MPI:
    DataReader              = ROOT.DataReaderMPI['ROOTDataReader'] # DataReaderMPI is a template; use [] to specify the type
    AmpToolsInterface       = ROOT.AmpToolsInterfaceMPI
else:
    DataReader              = ROOT.ROOTDataReader
    AmpToolsInterface       = ROOT.AmpToolsInterface
Zlm                         = ROOT.Zlm
ParameterManager            = ROOT.ParameterManager
MinuitMinimizationManager   = ROOT.MinuitMinimizationManager

############## LOAD CONFIGURATION FILE ##############
cfgfile = f'{REPO_HOME}/gen_amp/fit_res.cfg'
parser = ConfigFileParser(cfgfile)
cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
if RANK_MPI == 0:
    cfgInfo.display()

############## REGISTER OBJECTS FOR AMPTOOLS ##############
AmpToolsInterface.registerAmplitude( Zlm() )
AmpToolsInterface.registerDataReader( DataReader() ) 

ati = AmpToolsInterface( cfgInfo )
parMgr: ParameterManager = ati.parameterManager()

AmpToolsInterface.setRandomSeed(args.randomSeed)
runFits(args.numRnd)
