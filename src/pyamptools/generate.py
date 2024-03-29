import argparse

def gen_amp():
    ''' generate simulations using gen_amp '''
    generator = _setup_generator("gen_amp")
    generator.generate()

def gen_vec_ps():
    ''' generate simulations using gen_vec_ps '''
    generator = _setup_generator("gen_vec_ps")
    generator.generate()

def _setup_generator(generator_name):
    '''Setup the generator and its conditions'''
    # Parse command line args first in case help is requested

    conditions = _load_conditions(generator_name)

    import ROOT
    from pyamptools import atiSetup
    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), accelerator='cpu', use_genamp=True) # RANK_MPI defaults to 0 even if not using MPI
    generator_class = getattr(ROOT, generator_name)
    generator = generator_class()
    for k, v in conditions.items():
        setattr(generator, k, v)
    return generator

def _load_conditions(generator_name):
    parser = argparse.ArgumentParser(description="Description of your program")

    # Define command line arguments
    parser.add_argument('-c', '--configfile', default="", help='Config file')
    parser.add_argument('-o', '--outname', default=f"{generator_name}.root", help='ROOT file output name')
    parser.add_argument('-hd', '--hddmname', default="", help='HDDM file output name [optional]')
    parser.add_argument('-l', '--lowMass', type=float, default=0.2, help='Low edge of mass range (GeV) [optional]')
    parser.add_argument('-u', '--highMass', type=float, default=2.0, help='Upper edge of mass range (GeV) [optional]')
    parser.add_argument('-n', '--nEvents', type=int, default=10000, help='Minimum number of events to generate [optional]')
    parser.add_argument('-m', '--beamMaxE', type=float, default=12.0, help='Electron beam energy (or photon energy endpoint) [optional]')
    parser.add_argument('-p', '--beamPeakE', type=float, default=9.0, help='Coherent peak photon energy [optional]')
    parser.add_argument('-a', '--beamLowE', type=float, default=3.0, help='Minimum photon energy to simulate events [optional]')
    parser.add_argument('-b', '--beamHighE', type=float, default=12.0, help='Maximum photon energy to simulate events [optional]')
    parser.add_argument('-r', '--runNum', type=int, default=30731, help='Run number assigned to generated events [optional]')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random number seed initialization [optional]')
    parser.add_argument('-t', '--slope', type=float, default=6.0, help='Momentum transfer slope [optional]')
    parser.add_argument('-tmin', '--lowT', type=float, default=0.0, help='Minimum momentum transfer [optional]')
    parser.add_argument('-tmax', '--highT', type=float, default=12.0, help='Maximum momentum transfer [optional]')
    parser.add_argument('-d', '--diag', action='store_true', help='Plot only diagnostic histograms [optional]')
    parser.add_argument('-v', '--centeredVertex', action='store_false', help='Throw vertex distribution in gen_amp, not in hdgeant(4) [not recommended]')
    parser.add_argument('-f', '--genFlat', action='store_true', help='Generate flat in M(X) (no physics) [optional]')
    parser.add_argument('-fsroot', action='store_true', help='Enable output in FSRoot format')

    args = parser.parse_args()

    # Accessing the parsed arguments
    print("Config file:", args.configfile)
    print("ROOT file output name:", args.outname)
    print("HDDM file output name:", args.hddmname)
    print("Low edge of mass range (GeV):", args.lowMass)
    print("Upper edge of mass range (GeV):", args.highMass)
    print("Minimum number of events to generate:", args.nEvents)
    print("Electron beam energy (or photon energy endpoint):", args.beamMaxE)
    print("Coherent peak photon energy:", args.beamPeakE)
    print("Minimum photon energy to simulate events:", args.beamLowE)
    print("Maximum photon energy to simulate events:", args.beamHighE)
    print("Run number assigned to generated events:", args.runNum)
    print("Random number seed initialization:", args.seed)
    print("Momentum transfer slope:", args.slope)
    print("Minimum momentum transfer:", args.lowT)
    print("Maximum momentum transfer:", args.highT)
    print("Plot only diagnostic histograms:", args.diag)
    print("Throw vertex distribution in gen_amp, not in hdgeant(4):", args.centeredVertex)
    print("Generate flat in M(X) (no physics):", args.genFlat)
    print("Enable output in FSRoot format:", args.fsroot)

    return vars(args)
