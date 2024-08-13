import argparse
from multiprocessing import Pool
from time import time

from omegaconf import OmegaConf
from pyamptools import atiSetup
from pyamptools.utility.general import Silencer, glob_sort_captured
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

        ati.finalizeFit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Restructure normalization integrals')
    parser.add_argument('yaml', type=str, help='yaml file with paths')
    parser.add_argument('-n', '--ncores', type=int, default=1, help='number of cores to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    args = parser.parse_args()
    yaml = args.yaml
    yaml_file = OmegaConf.load(yaml)
    verbose = args.verbose

    output_directory = yaml_file["amptools"]["output_directory"]
    search_fmt = f"{output_directory}/bin_[]/*.cfg"
    cfgfiles = glob_sort_captured(search_fmt)

    if len(cfgfiles) == 0:
        print(f"\nNo cfg files found! Your search string should sort by value captured by brackets:\n  {search_fmt}\n")
        exit(0)

    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), verbose=verbose)

    # # ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude(Zlm())
    AmpToolsInterface.registerAmplitude(Vec_ps_refl())
    AmpToolsInterface.registerAmplitude(OmegaDalitz())
    AmpToolsInterface.registerAmplitude(BreitWigner())
    AmpToolsInterface.registerAmplitude(Piecewise())
    AmpToolsInterface.registerAmplitude(PhaseOffset())
    AmpToolsInterface.registerAmplitude(TwoPiAngles())
    AmpToolsInterface.registerAmplitude(Uniform())
    AmpToolsInterface.registerDataReader(DataReader())
    AmpToolsInterface.registerDataReader(DataReaderTEM())
    AmpToolsInterface.registerDataReader(DataReaderFilter())
    AmpToolsInterface.registerDataReader(DataReaderBootstrap()) 

    start_time = time()

    # ############################################################################
    # STEP 1) Ask AmpTools to dump ampvecs and normint to a ROOT file and a text file respectively
    if args.ncores > 1:
        print(f"\nProcessing {len(cfgfiles)} config files using {args.ncores} processes...\n")
        pool_args = [(cfgfile, verbose) for cfgfile in cfgfiles]
        with Pool(args.ncores) as p:
            p.map(extract_normint_ampvecs, pool_args)
    else:
        print(f"\nProcessing {len(cfgfiles)} config files...\n")
        for cfgfile in cfgfiles:
            extract_normint_ampvecs( (cfgfile, verbose) )
    print("\nAll config files have been processed!")

    ############################################################################
    # STEP 2) Restructure ampvecs and normints into arrays and save to pkl files
    print("\nRestructuring processed files into arrays and saving to pkl files")
    print("  This step cannot use multiple processes...")
    restructured_amps = restructure_amps(yaml_file, treename="kin")
    restructued_normints = restructure_normints(yaml_file)
    print(f"\nrElapsed time: {time() - start_time:.2f} seconds\n")

# intensityManager = ati.intensityManager(reactions[0].reactionName())

# amps = cfgInfo.amplitudeList()
# for ampinfo in amps:
#     print(intensityManager.productionFactor(ampinfo.fullName()))
#     print(float(intensityManager.getScale(ampinfo.fullName())))
#     print(f"{ampinfo.fullName()} {ampinfo.real()} {ampinfo.fixed()}")