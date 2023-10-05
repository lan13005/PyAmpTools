import ROOT
import os
import glob
from termcolor import colored

def test_check_root_found_functions():
    include_dirs = [
        f'{os.environ["REPO_HOME"]}/external/AMPTOOLS_AMPS',
        f'{os.environ["REPO_HOME"]}/external/AMPTOOLS_DATAIO',
        f'{os.environ["AMPTOOLS"]}',
        f'{os.environ["AMPPLOTTER"]}',
    ]

    src_files = []
    for include_dir in include_dirs:
        # print(f' =========== Processing {include_dir} =========== ')
        srcs = glob.glob(f'{include_dir}/**/*.cc') + glob.glob(f'{include_dir}/*.cc') # subdirectories + current directory
        srcs = [src.split('/')[-1].split('.')[0] for src in srcs]
        srcs = [src for src in srcs if 'GPU' not in src and 'MPI' not in src]
        src_files += srcs

    ############## LOAD LIBRARIES ##############
    ROOT.gSystem.Load('libAmps.so')
    ROOT.gSystem.Load('libDataIO.so')
    ROOT.gSystem.Load('libAmpTools.so')

    ROOT.initializeAmps(False)
    ROOT.initializeDataIO(False)

    for src in src_files:
        has_attribute = hasattr(ROOT, src)
        print( colored(f'{src} exists? {has_attribute}', 'green' if has_attribute else 'red') )
        assert( has_attribute ), f'{src} does not exist in ROOT, check library!'
