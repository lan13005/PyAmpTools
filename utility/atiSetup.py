import ROOT
from ROOT import pythonization
import os
from utils import check_shared_lib_exists, get_pid_family, check_nvidia_devices
from pythonization import pythonize_parMgr

kModule = 'atiSetup'

########################################################
#  This file is used to setup the amptools environment #
#  It is called by the user in their python script     #
#  and should be called before any other amptools      #
#  functions are called.                               #
########################################################

def setup(calling_globals, accelerator='mpigpu', use_fsroot=False, use_genamp=False):
    '''
    Performs basic setup, loading libraries and setting aliases

    Args:
        calling_globals (dict): globals() from the calling function
        accelerator (str): accelerator flag from argparse ~ ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi']
        use_fsroot (bool): True if FSRoot library should be loaded
        use_genamp (bool): True if GenAmp library should be loaded
    '''
    USE_MPI, USE_GPU, RANK_MPI = loadLibraries( accelerator, use_fsroot, use_genamp )
    set_aliases( calling_globals, USE_MPI, USE_GPU )

    return USE_MPI, USE_GPU, RANK_MPI

def loadLibraries(accelerator, use_fsroot=False, use_genamp=False):
    ''' Load all libraries '''
    USE_MPI, USE_GPU, RANK_MPI = prepare_mpigpu(accelerator)
    SUFFIX  = "_GPU" if USE_GPU else ""
    SUFFIX += "_MPI" if USE_MPI else ""

    if RANK_MPI == 0:
        print("\n------------------------------------------------")
        print(f'{kModule}| MPI is {"enabled" if USE_MPI else "disabled"}')
        print(f'{kModule}| GPU is {"enabled" if USE_GPU else "disabled"}')
        print("------------------------------------------------\n\n")
    #################### LOAD LIBRARIES (ORDER MATTERS!) ###################

    loadLibrary(f'libAmpTools{SUFFIX}.so', RANK_MPI)
    loadLibrary(f'libAmpPlotter.so', RANK_MPI)
    loadLibrary(f'libAmpsDataIO{SUFFIX}.so', RANK_MPI) # Depends on AmpPlotter!
    loadLibrary(f'libFSRoot.so', RANK_MPI, use_fsroot)
    loadLibrary(f'libAmpsGen.so', RANK_MPI, use_genamp)

    # Dummy functions that just prints "initialization"
    #  This is to make sure the libraries are loaded
    #  as python is interpreted.
    if RANK_MPI == 0: print("\n\n------------------------------------------------")
    ROOT.initialize( RANK_MPI == 0 )
    if use_fsroot: ROOT.initialize_fsroot( RANK_MPI == 0 )
    if RANK_MPI == 0: print("------------------------------------------------\n")

    return USE_MPI, USE_GPU, RANK_MPI

def loadLibrary(libName, RANK_MPI=0, IS_REQUESTED=True):
    ''' Load a shared library and print IS_REQUESTED '''
    statement = f'Loading library {libName} '
    libExists = check_shared_lib_exists(libName)
    if RANK_MPI == 0: print(f'{kModule}| {statement:.<45}', end='')
    if IS_REQUESTED:
        if libExists:
            ROOT.gSystem.Load(libName)
            status = "ON"
        else: status = 'NOT FOUND, SKIPPING'
    else: status = "OFF"
    if RANK_MPI == 0: print(f'  {status}')

def set_aliases( called_globals, USE_MPI, USE_GPU ):
    '''
    Due to MPI requiring c++ templates and the fact that all classes live under the ROOT namespace, aliasing can clean up the code significantly.
    A dictionary of aliases is appended to the globals() function of the calling function thereby making the aliases available in the calling function.

    Args:
        called_globals (dict): globals() from the calling function

    '''
    aliases = {
        ############### PyROOT RELATED ################
        'gInterpreter':               ROOT.gInterpreter,

        ############### AmpTools RELATED ##############
        'AmpToolsInterface':          ROOT.AmpToolsInterfaceMPI if USE_MPI else ROOT.AmpToolsInterface,
        'ConfigFileParser':           ROOT.ConfigFileParser,
        'ConfigurationInfo':          ROOT.ConfigurationInfo,
        'Zlm':                        ROOT.Zlm,
        'Vec_ps_refl':                ROOT.Vec_ps_refl,
        'OmegaDalitz':                ROOT.OmegaDalitz,
        'BreitWigner':                ROOT.BreitWigner,
        'Piecewise':                  ROOT.Piecewise,
        'PhaseOffset':                ROOT.PhaseOffset,
        'TwoPiAngles':                ROOT.TwoPiAngles,
        'ParameterManager':           ROOT.ParameterManager,
        'MinuitMinimizationManager':  ROOT.MinuitMinimizationManager,

        ############## DataReader RELATED ##############
        # DataReaderMPI is a template; use [] to specify the type
        'DataReader':                 ROOT.DataReaderMPI['ROOTDataReader'] if USE_MPI else ROOT.ROOTDataReader,
        'DataReaderTEM':              ROOT.DataReaderMPI['ROOTDataReaderTEM'] if USE_MPI else ROOT.ROOTDataReaderTEM,
        'DataReaderFilter':           ROOT.DataReaderMPI['ROOTDataReaderFilter'] if USE_MPI else ROOT.ROOTDataReaderFilter,
        'DataReaderBootstrap':        ROOT.DataReaderMPI['ROOTDataReaderBootstrap'] if USE_MPI else ROOT.ROOTDataReaderBootstrap,

        ########### PLOTTER / RESULTS RELATED ###########
        'FitResults':                 ROOT.FitResults,
        'EtaPiPlotGenerator':         ROOT.EtaPiPlotGenerator,
        'PlotGenerator':              ROOT.PlotGenerator,
        'TH1':                        ROOT.TH1,
        'TFile':                      ROOT.TFile,
        'AmplitudeInfo':              ROOT.AmplitudeInfo,
    }
    if USE_GPU:
        aliases['GPUManager'] = ROOT.GPUManager

    called_globals.update(aliases)

def prepare_mpigpu(accelerator):
    '''
    Sets environment variables to use MPI and/or GPU if requested.
        Checks who called the python script. If bash (single process). If mpiexec/mpirun (then MPI)

    Args:
        accelerator (str): accelerator flag from argparse ~ ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi']

    Returns:
        USE_MPI (bool): True if MPI is to be used
        USE_GPU (bool): True if GPU is to be used
        RANK_MPI (int): MPI rank of the process (0 by default even if MPI is not used)
    '''
    assert( accelerator in ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi'] ), f'Invalid accelerator flag: {accelerator}'
    called, parent = get_pid_family()
    print(f'{kModule}| {parent} called {called}')

    USE_MPI = False
    USE_GPU = False
    # mpiexec is executed on the leader node
    # orted, OpenMPI's daemon process, is executed on the worker nodes
    if ("mpi" in parent or parent == 'orted') and ('mpi' in accelerator):
        USE_MPI = True
    if (check_nvidia_devices()[0]) and ('gpu' in accelerator):
        USE_GPU = True

    ## SETUP ENVIRONMENT FOR MPI AND/OR GPU ##
    if USE_MPI:
        from mpi4py import rc as mpi4pyrc
        mpi4pyrc.threads = False
        mpi4pyrc.initialize = False
        from mpi4py import MPI
        MPI.Init()
        RANK_MPI = MPI.COMM_WORLD.Get_rank()
        SIZE_MPI = MPI.COMM_WORLD.Get_size()
        print(f'{kModule}| Found Task with Rank: {RANK_MPI} of {SIZE_MPI}')
        assert( (USE_MPI and (SIZE_MPI > 1)) )
    else:
        RANK_MPI = 0
        SIZE_MPI = 1

    return USE_MPI, USE_GPU, RANK_MPI
