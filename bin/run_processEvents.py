import argparse
from multiprocessing import Pool
from time import time

from pyamptools import atiSetup
from pyamptools.utility.general import Silencer, glob_sort_captured, load_yaml
from pyamptools.utility.restructure_amps import restructure_amps
from pyamptools.utility.restructure_normints import restructure_normints


def extract_normint_ampvecs(args):

    """
    calling ati.likelihood() (re)computes everything including the normalization integrals
    and all ampVecs (the complex amplitude storage arrays)

    ati.finalizeFit() dumps both of these into root files
    ROOT files are quite slow for reading into python, so one should probably convert them 
    to another file format
    """

    cfgfile, verbose = args
    print(f"Processing {cfgfile}")

    ############## LOAD CONFIGURATION FILE ######s########
    with Silencer(show_stderr=True, show_stdout=verbose):

        cfgparser = ConfigFileParser(cfgfile)
        cfgInfo: ConfigurationInfo = cfgparser.getConfigurationInfo()

        ati = AmpToolsInterface(cfgInfo)
        ati.likelihood()

        # This bit was needed because m_genMCVecs was not being allocated as it required a call
        #   to normIntInterface::forceCacheUpdate(). Fixed AmpToolsInterface::finalizeFit to do this
        # cfginfo = ati.configurationInfo()
        # reactions = cfginfo.reactionList() # [ ReactionInfo* ]
        # for reaction in reactions:
        #     normIntInterface = ati.normIntInterface(reaction.reactionName())
        #     normIntInterface.forceCacheUpdate(False)

        ati.finalizeFit("", True) # use positonal args not kwargs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Restructure normalization integrals')
    parser.add_argument('main_yaml', type=str, help='main yaml file')
    parser.add_argument('-np', '--n_processes', type=int, default=-1, help='number of processes to use')
    parser.add_argument('-ia', '--include_accmc', action='store_true', help='include accmc in ampvec root file')
    parser.add_argument('-ig', '--include_genmc', action='store_true', help='include genmc in ampvec root file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    args = parser.parse_args()
    main_yaml = args.main_yaml
    main_dict = load_yaml(main_yaml)
    verbose = args.verbose

    n_processes = args.n_processes
    if n_processes < 1:
        n_processes = main_dict["n_processes"]

    output_directory = main_dict["amptools"]["output_directory"]
    search_fmt = f"{output_directory}/bin_[]/bin_*.cfg"
    cfgfiles = glob_sort_captured(search_fmt)

    if len(cfgfiles) == 0:
        print(f"\nNo cfg files found! Your search string should sort by value captured by brackets:\n  {search_fmt}\n")
        exit(0)

    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), verbose=verbose)

    start_time = time()

    # ############################################################################
    # STEP 1) Ask AmpTools to dump ampvecs and normint to a ROOT file and a text file respectively
    _verbose = False # instead of CLI verbose. There is not much useful information dumped by AmpTools in this process so hard code a False
    if n_processes > 1:
        print(f"\nProcessing {len(cfgfiles)} config files using {n_processes} processes...\n")
        pool_args = [(cfgfile, _verbose) for cfgfile in cfgfiles]
        with Pool(n_processes) as p:
            p.map(extract_normint_ampvecs, pool_args)
    else:
        print(f"\nProcessing {len(cfgfiles)} config files...\n")
        for cfgfile in cfgfiles:
            extract_normint_ampvecs( (cfgfile, _verbose) )
    print("\nAll config files have been processed!")

    ############################################################################
    # STEP 2) Restructure ampvecs and normints into arrays and save to pkl files
    print("\nRestructuring processed files into arrays and saving to pkl files")
    print("  This step cannot use multiple processes...")
    restructured_amps = restructure_amps(main_dict, treename="kin", include_accmc=args.include_accmc, include_genmc=args.include_genmc)
    restructued_normints = restructure_normints(main_dict, verbose=verbose)
    print(f"\nElapsed time: {time() - start_time:.2f} seconds\n")

# intensityManager = ati.intensityManager(reactions[0].reactionName())

# amps = cfgInfo.amplitudeList()
# for ampinfo in amps:
#     print(intensityManager.productionFactor(ampinfo.fullName()))
#     print(float(intensityManager.getScale(ampinfo.fullName())))
#     print(f"{ampinfo.fullName()} {ampinfo.real()} {ampinfo.fixed()}")