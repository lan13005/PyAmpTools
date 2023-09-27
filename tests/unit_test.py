import os
import subprocess
import ROOT
import glob
from termcolor import colored

def test_environ_is_set():
	assert ( os.environ['REPO_HOME'] != "" )
	assert ( os.environ['AMPTOOLS_HOME'] != "" )
	assert ( os.environ['AMPPLOTTER'] != "" )
	assert ( os.environ['AMPTOOLS'] != "" )

def test_fit():
	REPO_HOME = os.environ['REPO_HOME']
	os.chdir(f'{REPO_HOME}/tests/samples')
	cmd=f"python {REPO_HOME}/EXAMPLES/python/fit.py -c fit_res.cfg"
	print(cmd)
	return_code = subprocess.call(cmd, shell=True)
	print(return_code)
	os.system(r'rm -f *.fit normint*') # clean up
	assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"

def test_check_root_found_functions():
    include_dirs = [
        f'{os.environ["REPO_HOME"]}/external/AMPTOOLS_AMPS',
        f'{os.environ["REPO_HOME"]}/external/AMPTOOLS_DATAIO',
        f'{os.environ["AMPTOOLS"]}',
        f'{os.environ["AMPPLOTTER"]}',
    ]

    not_implemented = [
        'GPUManager'
        ]

    src_files = []
    for include_dir in include_dirs:
        # print(f' =========== Processing {include_dir} =========== ')
        srcs = glob.glob(f'{include_dir}/**/*.cc') + glob.glob(f'{include_dir}/*.cc') # subdirectories + current directory
        srcs = [src.split('/')[-1].split('.')[0] for src in srcs]
        srcs = [src for src in srcs if src not in not_implemented]
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
