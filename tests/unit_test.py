import os
import subprocess
import ROOT
import glob
from termcolor import colored
from test_files import parMgr

def test_environ_is_set():
	assert ( os.environ['REPO_HOME'] != "" )
	assert ( os.environ['AMPTOOLS_HOME'] != "" )
	assert ( os.environ['AMPPLOTTER'] != "" )
	assert ( os.environ['AMPTOOLS'] != "" )

def test_parMgr():
    nProdPars, prefit_nll, par_real, par_imag, post_nll = parMgr.runTest()
    assert( nProdPars == 6 )
    assert( prefit_nll != 1e6 and prefit_nll is not None )
    assert( par_real == 15 and par_imag == 0 )
    assert( post_nll != 1e6 and post_nll is not None )

def test_AmpPars():
    REPO_HOME = os.environ["REPO_HOME"]
    base_dir = f'{REPO_HOME}/tests/test_files'
    fit_results = f'{base_dir}/result.fit'
    cmd = f'python {base_dir}/AmpPars.py {fit_results}'
    print(cmd)
    os.system(cmd)

def test_extract_ff():
   REPO_HOME = os.environ["REPO_HOME"]
   fit_results = f'{REPO_HOME}/tests/test_files/result.fit'
   output_file = f'{REPO_HOME}/tests/ff.txt'
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/extract_ff.py', fit_results, output_file, '-regex_merge', "'.*::(.*)::.*~>\\1'", "'.*(.)$~>\\1'"]
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_file) and os.path.getsize(output_file) > 0 )
   os.system(f'rm {output_file}')

def test_plotgen():
   REPO_HOME = os.environ["REPO_HOME"]
   fit_results = f'{REPO_HOME}/tests/test_files/result.fit'
   output_root_file = f'{REPO_HOME}/tests/plotgen_test.root'
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/plotgen.py', fit_results, '-o', output_root_file]
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_root_file) and os.path.getsize(output_root_file) > 1000 )
   os.system(f'rm {output_root_file}')

def test_fit():
	REPO_HOME = os.environ['REPO_HOME']
	cmd=f"python {REPO_HOME}/EXAMPLES/python/fit.py {REPO_HOME}/tests/samples/fit_res.cfg"
	print(cmd)
	return_code = subprocess.call(cmd, shell=True)
	print(return_code)
	os.system(r'rm -f *.fit normint*') # clean up
	assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"

def test_mcmc():
   REPO_HOME = os.environ['REPO_HOME']
   cfgfile = f'{REPO_HOME}/gen_amp/fit_res.cfg'
   mle_fit = f'{REPO_HOME}/tests/test_files/result.fit'
   ofolder = f'{REPO_HOME}/tests/mcmc'
   cmd=f"python {REPO_HOME}/EXAMPLES/python/mcmc.py {cfgfile} {mle_fit} -o {ofolder} -f 'mcmc.h5' -n 20 -b 10 -s 10 -overwrite"
   print(cmd)
   return_code = subprocess.call(cmd, shell=True)
   print(return_code)
   assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"
   os.system(r'rm -rf {ofolder}') # clean up

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
