import os

import ROOT

from pyamptools.utility.general import check_nvidia_devices, check_shared_lib_exists, get_pid_family
from pyamptools.utility.pythonization import pythonize_parMgr  # noqa

kModule = "atiSetup"

########################################################
#  This file is used to setup the amptools environment #
#  It is called by the user in their python script     #
#  and should be called before any other amptools      #
#  functions are called.                               #
########################################################


def setup(calling_globals, accelerator="mpigpu", use_fsroot=False, use_genamp=False, verbose=True):
    """
    Performs basic setup, loading libraries and setting aliases

    Args:
        calling_globals (dict): globals() from the calling function
        accelerator (str): accelerator flag from argparse ~ ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi']
        use_fsroot (bool): True if FSRoot library should be loaded
        use_genamp (bool): True if GenAmp library should be loaded
    """
    USE_MPI, USE_GPU, RANK_MPI = loadLibraries(accelerator, use_fsroot, use_genamp, verbose=verbose)
    set_aliases(calling_globals, USE_MPI, USE_GPU, use_fsroot, verbose=verbose)

    return USE_MPI, USE_GPU, RANK_MPI


def loadLibraries(accelerator, use_fsroot=False, use_genamp=False, verbose=True):
    """Load all libraries"""
    USE_MPI, USE_GPU, RANK_MPI = prepare_mpigpu(accelerator, verbose=verbose)
    SUFFIX = "_GPU" if USE_GPU else ""
    SUFFIX += "_MPI" if USE_MPI else ""

    if RANK_MPI == 0 and verbose:
        print("\n------------------------------------------------")
        print(f"{kModule}| MPI is {'enabled' if USE_MPI else 'disabled'}")
        print(f"{kModule}| GPU is {'enabled' if USE_GPU else 'disabled'}")
        print("------------------------------------------------\n\n")
    #################### LOAD LIBRARIES (ORDER MATTERS!) ###################

    loadLibrary(f"libIUAmpTools{SUFFIX}.so", RANK_MPI, verbose=verbose)
    loadLibrary(f"libAmpTools{SUFFIX}.so", RANK_MPI, verbose=verbose)
    loadLibrary("libAmpPlotter.so", RANK_MPI, verbose=verbose)
    loadLibrary(f"libAmpsDataIO{SUFFIX}.so", RANK_MPI, verbose=verbose)  # Depends on AmpPlotter!
    loadLibrary("libFSRoot.so", RANK_MPI, use_fsroot, verbose=verbose)
    loadLibrary("libAmpsGen.so", RANK_MPI, use_genamp, verbose=verbose)

    # Dummy functions that just prints "initialization"
    #  This is to make sure the libraries are loaded
    #  as python is interpreted.
    if RANK_MPI == 0 and verbose:
        print("\n\n------------------------------------------------")
    ROOT.initialize(False)  # (RANK_MPI == 0)
    if use_fsroot:
        ROOT.initialize_fsroot(False)  # (RANK_MPI == 0)
    if RANK_MPI == 0 and verbose:
        print("------------------------------------------------\n")

    return USE_MPI, USE_GPU, RANK_MPI


def loadLibrary(libName, RANK_MPI=0, IS_REQUESTED=True, verbose=True):
    """Load a shared library and print IS_REQUESTED"""
    statement = f"Loading library {libName} "
    libExists = check_shared_lib_exists(libName)
    if RANK_MPI == 0 and verbose:
        print(f"{kModule}| {statement:.<45}", end="")
    if IS_REQUESTED:
        if libExists:
            ROOT.gSystem.Load(libName)
            status = "ON"
        else:
            status = "NOT FOUND, SKIPPING"
    else:
        status = "OFF"
    if RANK_MPI == 0 and verbose:
        print(f"  {status}")

def get_linked_objects_to_alias():
    
    """
    Helper function for set_aliases. Searches for objects in FSROOT + AmpTools shared libraries Linkdef.h files
    
    Returns:
        Dict[str, List[str]]: keys = full path location to Linkdef.h file, values = list of objects to alias
    """
    
    PYAMPTOOLS_HOME = os.getenv("PYAMPTOOLS_HOME")
    
    if os.path.exists(f"{PYAMPTOOLS_HOME}/src/pyamptools/.aliases.txt"):
        with open(f"{PYAMPTOOLS_HOME}/src/pyamptools/.aliases.txt", "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines]

    # PyROOT dumps a Linkdef.h file when it binds objects from a shared library
    #   we can parse it to get the objects to alias
    #   Currently only care about FSROOT and AmpTools
    linkdef_files = []
    external_dir = os.path.join(PYAMPTOOLS_HOME, "external")
    for root, dirs, files in os.walk(external_dir):
        if "root" in root or "GENERATORS" in root:
            continue
        for file in files:
            if "MPI" in file or "GPU" in file:
                continue
            if file.endswith("Linkdef.h"):
                linkdef_files.append(os.path.join(root, file))
                
    # Loop through Linkdef files and extract the bound objects
    #   Ignore specific keys to handle later, for instance MPI needs to manually set as it requires a different setup
    objects_to_alias = {}
    ignore_keys = ["dict", "LING", 'initialize']
    for linkdef_file in linkdef_files:
        objects_to_alias[linkdef_file] = []
        with open(linkdef_file, "r") as symbols:
            for line in symbols:
                line = line.strip().strip("#pragma link C++ defined_in").split("/")[-1].split('.h')[0].strip('"')
                if any(key in line for key in ignore_keys) or len(line) == 0:
                    continue
                if line.startswith("AmpToolsInterface") or line.startswith("DataReader"): # handle later
                    continue
                objects_to_alias[linkdef_file].append(line)
                
    # Dump the objects to a file to avoid re-parsing if exists
    flat_objects_to_alias = []
    with open(f"{PYAMPTOOLS_HOME}/src/pyamptools/.aliases.txt", "w") as f:
        for _, objects in objects_to_alias.items():
            flat_objects_to_alias.extend(objects)
            for object in objects:
                f.write(f"{object}\n")
                
    return flat_objects_to_alias

def set_aliases(called_globals, USE_MPI, USE_GPU, use_fsroot, verbose=False):
    """
    Due to MPI requiring c++ templates and the fact that all classes live under the ROOT namespace, aliasing can clean up the code significantly.
    A dictionary of aliases is appended to the globals() function of the calling function thereby making the aliases available in the calling function.

    Args:
        called_globals (dict): globals() from the calling function
    """
    
    aliases = {}
    
    for obj in get_linked_objects_to_alias():
        if not use_fsroot and obj.startswith("FS"):
            continue
        try:
            aliases[obj] = getattr(ROOT, obj)
        except AttributeError:
            if verbose:
                print(f"{kModule}| minor warning: Unable to alias {obj} - doesn't exist under ROOT namespace")
            pass
    
    aliases.update({
        ########### MANUALLY HANDLE SPECIAL CASES ############
        "gInterpreter": ROOT.gInterpreter,
        "AmpToolsInterface": ROOT.AmpToolsInterfaceMPI if USE_MPI else ROOT.AmpToolsInterface,
        "DataReader": ROOT.DataReaderMPI["ROOTDataReader"] if USE_MPI else ROOT.ROOTDataReader,
        "DataReaderTEM": ROOT.DataReaderMPI["ROOTDataReaderTEM"] if USE_MPI else ROOT.ROOTDataReaderTEM,
        "DataReaderFilter": ROOT.DataReaderMPI["ROOTDataReaderFilter"] if USE_MPI else ROOT.ROOTDataReaderFilter,
        "DataReaderBootstrap": ROOT.DataReaderMPI["ROOTDataReaderBootstrap"] if USE_MPI else ROOT.ROOTDataReaderBootstrap,
    })
    if USE_GPU:
        aliases["GPUManager"] = ROOT.GPUManager

    called_globals.update(aliases)
    register_amps_dataio(called_globals)


def register_amps_dataio(globals):
    """REGISTER OBJECTS FOR AMPTOOLS"""
    globals["AmpToolsInterface"].registerAmplitude(globals["Zlm"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["Vec_ps_refl"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["OmegaDalitz"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["BreitWigner"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["Piecewise"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["PhaseOffset"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["TwoPiAngles"]())
    globals["AmpToolsInterface"].registerAmplitude(globals["Uniform"]())
    globals["AmpToolsInterface"].registerDataReader(globals["DataReader"]())
    globals["AmpToolsInterface"].registerDataReader(globals["DataReaderTEM"]())
    globals["AmpToolsInterface"].registerDataReader(globals["DataReaderFilter"]())
    globals["AmpToolsInterface"].registerDataReader(globals["DataReaderBootstrap"]())


def prepare_mpigpu(accelerator, verbose=True):
    """
    Sets environment variables to use MPI and/or GPU if requested.
        Checks who called the python script. If bash (single process). If mpiexec/mpirun (then MPI)

    Args:
        accelerator (str): accelerator flag from argparse ~ ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi']

    Returns:
        USE_MPI (bool): True if MPI is to be used
        USE_GPU (bool): True if GPU is to be used
        RANK_MPI (int): MPI rank of the process (0 by default even if MPI is not used)
    """

    if accelerator not in ["cpu", "mpi", "gpu", "mpigpu", "gpumpi"]:
        if ":" in accelerator:
            if verbose:
                print(f"{kModule}|  accelerator might be in SLURM form 'accelerator:device'. Attempting use of only the initial accelerator part")
            accelerator = accelerator.split(":")[0]
            if accelerator not in ["cpu", "mpi", "gpu", "mpigpu", "gpumpi"]:
                raise ValueError(f"{kModule}| Unable to parse remaining accelerator flag: {accelerator}")
        else:
            raise ValueError(f"{kModule}| accelerator flag: {accelerator}, must be one of ['cpu', 'mpi', 'gpu', 'mpigpu', 'gpumpi']")

    called, parent = get_pid_family()
    if verbose:
        print(f"{kModule}| {parent} called {called}")

    USE_MPI = False
    USE_GPU = False
    # mpiexec is executed on the leader node
    # orted, OpenMPI's daemon process, is executed on the worker nodes
    if ("mpi" in parent or parent == "orted") and ("mpi" in accelerator):
        USE_MPI = True
    if (check_nvidia_devices()[0]) and ("gpu" in accelerator):
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
        if verbose:
            print(f"{kModule}| Found Task with Rank: {RANK_MPI} of {SIZE_MPI}")
        assert USE_MPI and (SIZE_MPI > 1)
    else:
        RANK_MPI = 0
        SIZE_MPI = 1

    return USE_MPI, USE_GPU, RANK_MPI
